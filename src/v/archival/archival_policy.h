/*
 * Copyright 2021 Redpanda Data, Inc.
 *
 * Licensed as a Redpanda Enterprise file under the Redpanda Community
 * License (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 * https://github.com/redpanda-data/redpanda/blob/master/licenses/rcl.md
 */

#pragma once

#include "archival/probe.h"
#include "archival/types.h"
#include "cloud_storage/partition_manifest.h"
#include "cloud_storage/types.h"
#include "model/fundamental.h"
#include "storage/fwd.h"
#include "storage/ntp_config.h"

#include <seastar/core/io_priority_class.hh>
#include <seastar/core/rwlock.hh>

namespace archival {

enum class compacted_candidate_creation_error {
    no_segments,
    begin_offset_seek_error,
    end_offset_seek_error,
    offset_inside_batch,
    upload_size_unchanged,
    cannot_replace_manifest_entry,
};

std::ostream& operator<<(std::ostream&, compacted_candidate_creation_error);

enum class non_compacted_candidate_creation_error {
    no_segment_for_begin_offset,
    missing_ntp_config,
    failed_to_get_file_range,
    zero_content_length,
};

std::ostream& operator<<(std::ostream&, non_compacted_candidate_creation_error);

template<segment_upload_kind upload_kind>
struct error_for_upload_kind;

template<>
struct error_for_upload_kind<segment_upload_kind::compacted> {
    using type = compacted_candidate_creation_error;
};

template<>
struct error_for_upload_kind<segment_upload_kind::non_compacted> {
    using type = non_compacted_candidate_creation_error;
};

template<auto upload_kind>
using error_for_upload_kind_t = error_for_upload_kind<upload_kind>::type;

using candidate_creation_error = std::variant<
  compacted_candidate_creation_error,
  non_compacted_candidate_creation_error>;

std::ostream& operator<<(std::ostream&, const candidate_creation_error&);

ss::log_level log_level_for_error(const candidate_creation_error& error);

struct upload_candidate {
    segment_name exposed_name;
    model::offset starting_offset;
    size_t file_offset;
    size_t content_length;
    model::offset final_offset;
    size_t final_file_offset;
    model::timestamp base_timestamp;
    model::timestamp max_timestamp;
    model::term_id term;
    std::vector<ss::lw_shared_ptr<storage::segment>> sources;
    std::vector<cloud_storage::remote_segment_path> remote_sources;

    friend std::ostream& operator<<(std::ostream& s, const upload_candidate& c);
};

struct upload_candidate_with_locks {
    upload_candidate candidate;
    std::vector<ss::rwlock::holder> read_locks;
};

/// Wraps an error with an offset range, so that no
/// further upload candidates are created from this offset range.
template<auto upload_kind>
struct skip_offset_range {
    model::offset begin;
    model::offset end;
    error_for_upload_kind_t<upload_kind> error;

    friend std::ostream&
    operator<<(std::ostream& os, const skip_offset_range& skip_range) {
        fmt::print(
          os,
          "skip_offset_range{{begin: {}, end: {},error: {}}}",
          skip_range.begin,
          skip_range.end,
          skip_range.error);
        return os;
    }
};

template<segment_upload_kind upload_kind>
using candidate_creation_result = std::variant<
  upload_candidate_with_locks,
  skip_offset_range<upload_kind>,
  error_for_upload_kind_t<upload_kind>,
  std::monostate>;

using compacted_candidate_creation_result
  = candidate_creation_result<segment_upload_kind::compacted>;

using non_compacted_candidate_creation_result
  = candidate_creation_result<segment_upload_kind::non_compacted>;

/// Archival policy is responsible for extracting segments from
/// log_manager in right order.
///
/// \note It doesn't store a reference to log_manager or any segments
/// but uses ntp as a key to extract the data when needed.
class archival_policy {
public:
    explicit archival_policy(
      model::ntp ntp,
      std::optional<segment_time_limit> limit = std::nullopt,
      ss::io_priority_class io_priority = ss::default_priority_class());

    /// \brief regurn next upload candidate
    ///
    /// \param begin_inclusive is an inclusive begining of the range
    /// \param end_exclusive is an exclusive end of the range
    /// \param lm is a log manager
    /// \return initializd struct on success, empty struct on failure
    ss::future<non_compacted_candidate_creation_result> get_next_candidate(
      model::offset begin_inclusive,
      model::offset end_exclusive,
      ss::shared_ptr<storage::log>,
      const storage::offset_translator_state&,
      ss::lowres_clock::duration segment_lock_duration);

    ss::future<compacted_candidate_creation_result> get_next_compacted_segment(
      model::offset begin_inclusive,
      ss::shared_ptr<storage::log> log,
      const cloud_storage::partition_manifest& manifest,
      ss::lowres_clock::duration segment_lock_duration);

    static bool eligible_for_compacted_reupload(const storage::segment&);

    template<segment_upload_kind upload_kind>
    ss::future<candidate_creation_result<upload_kind>> next_upload_candidate(
      model::offset begin_inclusive,
      model::offset end_exclusive,
      ss::shared_ptr<storage::log> log,
      const storage::offset_translator_state& ot_state,
      const cloud_storage::partition_manifest& manifest,
      ss::lowres_clock::duration segment_lock_duration) {
        switch (upload_kind) {
        case segment_upload_kind::compacted:
            co_return co_await get_next_compacted_segment(
              begin_inclusive, log, manifest, segment_lock_duration);
        case segment_upload_kind::non_compacted:
            co_return co_await get_next_candidate(
              begin_inclusive,
              end_exclusive,
              log,
              ot_state,
              segment_lock_duration);
        }
    }

private:
    /// Check if the upload have to be forced due to timeout
    ///
    /// If the upload is idle longer than expected the next call to
    /// `get_next_candidate` will return partial result which will
    /// result in partial upload.
    bool upload_deadline_reached();

    struct lookup_result {
        ss::lw_shared_ptr<storage::segment> segment;
        const storage::ntp_config* ntp_conf;
        bool forced;
    };

    lookup_result find_segment(
      model::offset last_offset,
      model::offset adjusted_lso,
      ss::shared_ptr<storage::log>,
      const storage::offset_translator_state&);

    model::ntp _ntp;
    std::optional<segment_time_limit> _upload_limit;
    std::optional<ss::lowres_clock::time_point> _upload_deadline;
    ss::io_priority_class _io_priority;
};

} // namespace archival
