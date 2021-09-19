// Copyright (c) 2021 Hartmut Kaiser
// Copyright (c) 2021 Nanmiao Wu
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/algorithm.hpp>
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/collectives.hpp>

#include <cstddef>
#include <utility>
#include <vector>

using namespace hpx::collectives;

///////////////////////////////////////////////////////////////////////////////
constexpr char const* channel_communicator_name =
"lalalalala_hpx_hpx";

// the number of times
constexpr int times = 2;
////////////////////////////////////////////////////////////////////////

int hpx_main()
{
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
    std::uint32_t this_locality = hpx::get_locality_id();

    // allocate channel communicator
    auto comm = create_channel_communicator(hpx::launch::sync,
        channel_communicator_name, num_sites_arg(num_localities),
        this_site_arg(this_locality));

    std::uint32_t next_locality = (this_locality + 1) % num_localities;

    std::vector<int> msg_vec = {0, 1};

    int cnt = 0;

    int msg = msg_vec[this_locality];
      
    // send values to another locality
    set(comm, that_site_arg(next_locality), msg).get();

    auto got_msg = get<int>(comm, that_site_arg(next_locality));

    while(cnt < times)
    {
        auto done_msg = got_msg.then([&](auto && f) {
            int rec_msg = f.get();
            std::cout <<"Time: " << cnt 
                      << ", Locality " << this_locality << " received msg: " 
                      << rec_msg << "\n";
            // change msg by adding 10
            rec_msg += 10;

            // start next round
            set(comm, that_site_arg(next_locality), rec_msg).get();
            got_msg = get<int>(comm, that_site_arg(next_locality));

        });

        done_msg.get();
        cnt += 1;
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::init_params params;
    params.cfg = {"--hpx:run-hpx-main"};
    return hpx::init(argc, argv, params);
}

