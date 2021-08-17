//  Copyright (c)      2021 Nanmiao Wu
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include "../core/core.h"

#include <tuple>

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include "hpx/hpx.hpp"
#include "hpx/hpx_init.hpp"
#include "hpx/local/chrono.hpp"
#include "hpx/modules/collectives.hpp"

constexpr char const* channel_communicator_basename =
  "hpx_block_channel_communicator";


namespace detail{

  std::tuple<long, long, long> tile(std::uint32_t loc_idx, long column, std::uint32_t numlocs)
  {      
    long first_point;
    long n_points = column / numlocs;
    long rest = column % numlocs;
    if (loc_idx < rest){
      n_points ++;
    }

    if (rest != 0 && loc_idx >= rest){
      first_point = (n_points + 1) * rest + n_points * (loc_idx - rest);
    } else {
      first_point = n_points * loc_idx;
    }
    long last_point = first_point + n_points - 1;
    return std::make_tuple(first_point, last_point, n_points);
  }

  void assign_tasks(long first_point, long last_point, long n_inputs, 
      std::vector<std::vector<std::vector<char> > > inputs, 
      std::vector<std::vector<char> > &outputs, long dset,
      std::vector<std::vector<std::vector<std::pair<long, long> > > > dependencies,
      std::vector<std::vector<std::vector<std::pair<long, long> > > > reverse_dependencies)
  {
    std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);

    // for each site, create new channel_communicator
    std::vector<hpx::collectives::channel_communicator> comms;
    // need to know number of tasks first?
    comms.reserve(num_localities);

    // not general 
    for (std::size_t i = 0; i != num_localities; ++i)
    {
        comms.push_back(hpx::collectives::create_channel_communicator(hpx::launch::sync,
            channel_communicator_basename,
            num_sites_arg(num_localities),
            this_site_arg(i)));
    }

    std::vector<hpx::future<void>> tasks;
    tasks.reserve(num_localities);

    for (std::size_t i = 0; i != num_localities; ++i)
    {
        std::cout << "task: " << i << '\n';
        tasks.push_back(hpx::async(
            each_time_get_set_comm, comms[i], 
            long first_point, long last_point, long n_inputs, 
            std::vector<std::vector<std::vector<char> > > inputs, 
            std::vector<std::vector<char> > &outputs, long dset,
            std::vector<std::vector<std::vector<std::pair<long, long> > > > dependencies,
            std::vector<std::vector<std::vector<std::pair<long, long> > > > reverse_dependencies));
    }
    hpx::wait_all(tasks);

  }

  void each_time_get_set_comm(hpx::collectives::channel_communicator comm,
      long first_point, long last_point, long n_inputs, 
      std::vector<std::vector<std::vector<char> > > inputs, 
      std::vector<std::vector<char> > &outputs, long dset,
      std::vector<std::vector<std::vector<std::pair<long, long> > > > dependencies,
      std::vector<std::vector<std::vector<std::pair<long, long> > > > reverse_dependencies)
  {
    long offset = graph.offset_at_timestep(timestep);
    long width = graph.width_at_timestep(timestep);

    long last_offset = graph.offset_at_timestep(timestep-1);
    long last_width = graph.width_at_timestep(timestep-1);
    
    auto &deps = dependencies[dset];
    auto &rev_deps = reverse_dependencies[dset];

    std::vector<std::size_t> point_n_inputs_future_vec;
    std::vector<std::size_t> point_inputs_future_vec;

    for (long point = first_point; point <= last_point; ++point) {

      long point_index = point - first_point;

      auto &point_inputs = inputs[point_index];
      auto &point_n_inputs = n_inputs[point_index];
      auto &point_output = outputs[point_index];

      auto &point_deps = deps[point_index];
      auto &point_rev_deps = rev_deps[point_index];

      using data_type = std::vector<char>;

      std::vector<hpx::future<data_type>> gets;
      std::vector<hpx::future<void>> sets;

      //Receive 
      point_n_inputs = 0;
      if (point >= offset && point < offset + width) {
        for (auto interval : point_deps) {
          for (long dep = interval.first; dep <= interval.second; ++dep) {
            if (dep < last_offset || dep >= last_offset + last_width) {
              continue;
            }

            // Use shared memory for on-node data.
            if (first_point <= dep && dep <= last_point) {
              auto &output = outputs[dep - first_point];
              point_inputs[point_n_inputs].assign(output.begin(), output.end());
                  
            } else {
              //MPI_Irecv(point_inputs[point_n_inputs].data(),
              //          point_inputs[point_n_inputs].size(), MPI_BYTE,
              //          rank_by_point[dep], tag, MPI_COMM_WORLD, &req);
              gets.push_back(hpx::collectives::get<data_type>(comm, hpx::collectives::that_site_arg(locality_by_point[dep])));
              point_n_inputs_future_vec.push_back(point_n_inputs);
              point_inputs_future_vec.push_back(point_index);
            }
            point_n_inputs++;
          }
        }
      }

      // Send
      if (point >= last_offset && point < last_offset + last_width) {
        for (auto interval : point_rev_deps) {
          for (long dep = interval.first; dep <= interval.second; dep++) {
            if (dep < offset || dep >= offset + width || (first_point <= dep && dep <= last_point)) {
              continue;
            }
            //MPI_Isend(point_output.data(), point_output.size(), MPI_BYTE,
            //          rank_by_point[dep], tag, MPI_COMM_WORLD, &req);
            sets.push_back(hpx::collectives::set(comm, hpx::collectives::that_site_arg(locality_by_point[dep]), point_output));
          }
        }
      }
    }

    hpx::wait_all(sets, gets);
    for (std::size_t i = 0; i != gets.size(); ++i) {
      auto point_n_inputs = point_n_inputs_future_vec[i];
      auto point_idx = point_inputs_future_vec[i];
      auto rec_point_inputs = inputs[point_idx];
      rec_point_inputs[point_n_inputs]= gets[i].get();
    }

  }

  void execute_point(long timestep, long first_point, long last_point, long n_inputs, 
      std::vector<std::vector<std::vector<char> > > inputs, 
      std::vector<std::vector<const char *> > input_ptr,
      std::vector<std::vector<size_t> > input_bytes,
      std::vector<std::vector<char> > outputs,
      size_t scratch_bytes,
      char *scratch_ptr)
  {
    long offset = graph.offset_at_timestep(timestep);
    long width = graph.width_at_timestep(timestep);

    for (long point = std::max(first_point, offset); point <= std::min(last_point, offset + width - 1); ++point) {
      long point_index = point - first_point;

      auto &point_input_ptr = input_ptr[point_index];
      auto &point_input_bytes = input_bytes[point_index];
      auto &point_n_inputs = n_inputs[point_index];
      auto &point_output = outputs[point_index];

      graph.execute_point(timestep, point,
                          point_output.data(), point_output.size(),
                          point_input_ptr.data(), point_input_bytes.data(), point_n_inputs,
                          scratch_ptr + scratch_bytes * point_index, scratch_bytes);
    }   

  }
} // namespace detail

int hpx_main(int argc, char *argv[]) 
{    
  // get number of localities and this locality
  std::uint32_t num_localities = hpx::get_num_localities(hpx::launch::sync);
  std::uint32_t this_locality = hpx::get_locality_id();

  App app(argc, argv);
  if (this_locality == 0) app.display();

  std::vector<std::vector<char> > scratch;

  for (auto graph : app.graphs) {
    long first_point = this_locality * graph.max_width / num_localities;
    long last_point = (this_locality + 1) * graph.max_width / num_localities - 1;
    long n_points = last_point - first_point + 1;

    size_t scratch_bytes = graph.scratch_bytes_per_task;
    scratch.emplace_back(scratch_bytes * n_points);
    TaskGraph::prepare_scratch(scratch.back().data(), scratch.back().size());
  }
  
  double elapsed = 0.0;
  for (int iter = 0; iter < 1; ++iter) {
    
    hpx::chrono::high_resolution_timer timer;

    for (auto graph : app.graphs) {
   
      long first_point = this_locality * graph.max_width / num_localities;
      long last_point = (this_locality + 1) * graph.max_width / num_localities - 1;
      long n_points = last_point - first_point + 1;

      size_t scratch_bytes = graph.scratch_bytes_per_task;
      char *scratch_ptr = scratch[graph.graph_index].data();

      std::vector<int> locality_by_point(graph.max_width);
      for (int r = 0; r < num_localities; ++r) {
        long r_first_point = r * graph.max_width / num_localities;
        long r_last_point = (r + 1) * graph.max_width / num_localities - 1;
        for (long p = r_first_point; p <= r_last_point; ++p) {
          locality_by_point[p] = r;
        }
      }

      long max_deps = 0;
      for (long dset = 0; dset < graph.max_dependence_sets(); ++dset) {
        for (long point = first_point; point <= last_point; ++point) {
          long deps = 0;
          for (auto interval : graph.dependencies(dset, point)) {
            deps += interval.second - interval.first + 1;
          }
          max_deps = std::max(max_deps, deps);
        }
      }

      // Create input and output buffers.
      std::vector<std::vector<std::vector<char> > > inputs(n_points);
      std::vector<std::vector<const char *> > input_ptr(n_points);
      std::vector<std::vector<size_t> > input_bytes(n_points);
      std::vector<long> n_inputs(n_points);
      std::vector<std::vector<char> > outputs(n_points);
      for (long point = first_point; point <= last_point; ++point) {
        long point_index = point - first_point;

        auto &point_inputs = inputs[point_index];
        auto &point_input_ptr = input_ptr[point_index];
        auto &point_input_bytes = input_bytes[point_index];

        point_inputs.resize(max_deps);
        point_input_ptr.resize(max_deps);
        point_input_bytes.resize(max_deps);

        for (long dep = 0; dep < max_deps; ++dep) {
          point_inputs[dep].resize(graph.output_bytes_per_task);
          point_input_ptr[dep] = point_inputs[dep].data();
          point_input_bytes[dep] = point_inputs[dep].size();
        }

        auto &point_outputs = outputs[point_index];
        point_outputs.resize(graph.output_bytes_per_task);
      }

      // Cache dependencies.
      std::vector<std::vector<std::vector<std::pair<long, long> > > > dependencies(graph.max_dependence_sets());
      std::vector<std::vector<std::vector<std::pair<long, long> > > > reverse_dependencies(graph.max_dependence_sets());
      for (long dset = 0; dset < graph.max_dependence_sets(); ++dset) {
        dependencies[dset].resize(n_points);
        reverse_dependencies[dset].resize(n_points);

        for (long point = first_point; point <= last_point; ++point) {
          long point_index = point - first_point;

          dependencies[dset][point_index] = graph.dependencies(dset, point);
          reverse_dependencies[dset][point_index] = graph.reverse_dependencies(dset, point);
        }
      }
  
      for (long timestep = 0; timestep < graph.timesteps; ++timestep) {

        assign_tasks(long first_point, long last_point, long n_inputs, 
            std::vector<std::vector<std::vector<char> > > inputs, 
            std::vector<std::vector<char> > &outputs, long dset,
            std::vector<std::vector<std::vector<std::pair<long, long> > > > dependencies,
            std::vector<std::vector<std::vector<std::pair<long, long> > > > reverse_dependencies);
        
        execute_point(long timestep, long first_point, long last_point, long n_inputs, 
            std::vector<std::vector<std::vector<char> > > inputs, 
            std::vector<std::vector<const char *> > input_ptr,
            std::vector<std::vector<size_t> > input_bytes,
            std::vector<std::vector<char> > outputs,
            size_t scratch_bytes,
            char *scratch_ptr);       
      }    


    }
  
    elapsed = timer.elapsed(); 
    std::cout << "-----------------------------" << std::endl;
  }



  if (this_locality == 0) {
    app.report_timing(elapsed);
  }

  return hpx::finalize();;
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX, this example requires to run hpx_main on all
    // localities
    std::vector<std::string> const cfg = {
        "hpx.run_hpx_main!=1",
        "--hpx:ini=hpx.commandline.allow_unknown!=1",
        "--hpx:init=hpx.commandline.aliasing!=0"
    };
    hpx::init_params init_args;
    init_args.cfg = cfg;  
    
    return hpx::init(argc, argv, init_args);
}
#endif