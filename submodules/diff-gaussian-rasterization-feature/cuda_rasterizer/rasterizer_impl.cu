/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <cstdio>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"




// Helper function to find the next-highest bit of the MSB
// on the CPU.
// 查找most significant bit, 輸入變量n表示tile編號最大值x,y的乘積
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}





// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}



// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets, // inclusive sum of numbr of rendered gaussians
	uint64_t* gaussian_keys_unsorted, // outpuit
	uint32_t* gaussian_values_unsorted, //output
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
// 針對每一個tile決定其 key list 的開始與結束位置。
// L num rendered
// 決定point_list_keys 在哪一個tile開始與結束
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}






// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum <<<(P + 255) / 256, 256 >>> (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}







CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P) // P gaussians number
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128); // radius pointer
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.semantic_feature, P * NUM_SEMANTIC_CHANNELS, 128);
	obtain(chunk, geom.score_feature, P , 128);
	obtain(chunk, geom.tiles_touched, P, 128); // record the touched tiles for each gaussian
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128); // tmp for Inclusivesum
	obtain(chunk, geom.point_offsets, P, 128); // inclusive sum offset for tiles touched
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N) // N: image pixels width * height
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128); // tile_grid.x * tile_grid.y * sizeof(uint2))
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P) // P num_rendered tiles
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128); // value: gaussian id
	obtain(chunk, binning.point_list_unsorted, P, 128); // vlaue: gaussian id
	obtain(chunk, binning.point_list_keys, P, 128); // key: tile|depth
	obtain(chunk, binning.point_list_keys_unsorted, P, 128); //key: tile:depth
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128); // tmp sorting space
	return binning;
}







// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* semantic_feature, 
	const float* score_feature, ////
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color, // output 
	float* out_feature_map, // output 
	float* out_score_map, // output 
	float* out_depth, // output 
	int* radii, // output 
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);


	// create geonetry state
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P); 


	// make radius
	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}


	// create block, tile_grid
	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	
	
	// Dynamically resize image-based auxiliary buffers during training
	// create image state
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	
	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}


	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii, // o radius
		geomState.means2D, //o xy coords
		geomState.depths, //o 
		geomState.cov3D, //o 
		geomState.rgb, //o  2d means color
		geomState.conic_opacity,// output conic->used for compute prob
		tile_grid, // tile total number (x,y)
		geomState.tiles_touched, // o tiles touched sum array
		prefiltered
	), debug)


	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)


	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered; // total rendered number in pixels
	// geomState.point_offsets 存每個gaussians碰到的tile總數
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);


	// create bining state
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);


	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// P: number of gaussians
	duplicateWithKeys <<<(P + 255) / 256, 256 >>> (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets, // inclusive sum for rendered gaussians
		binningState.point_list_keys_unsorted, // o
		binningState.point_list_unsorted, // o
		radii,
		tile_grid
	)
	CHECK_CUDA(, debug)

	// point_list_keys_unsorted被tile總數所限制
	int bit = getHigherMsb(tile_grid.x * tile_grid.y); // get the fighest bit position


	// Sort complete list of (duplicated) Gaussian indices by keys
	// sort gaussian total list
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)


	// 製作一個array, tile_x* tile_y* 2, 紀錄每個tile在point lsit中的開始與結束位置
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);



	// Identify start and end of per-tile workloads in sorted list
	// 紀錄每個tile在point lsit中的開始與結束位置
	if (num_rendered > 0)
		identifyTileRanges <<<(num_rendered + 255) / 256, 256 >>> (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)



	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, 
		height,
		geomState.means2D,
		feature_ptr, // rgb
		semantic_feature, // semantic
		score_feature, // score
		geomState.depths, // depth
		geomState.conic_opacity, // conic + opacity
		imgState.accum_alpha, // output
		imgState.n_contrib, // output
		background,
		out_color, // output
		out_feature_map, // output
		out_score_map, 
		out_depth // output
	), debug)

	return num_rendered;
}









// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, 
	int R, // num_rendered
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* semantic_feature, 
	const float* score_feature, // score
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix, // input color
	const float* dL_dfeaturepix, // input semantic
	const float* dL_dscorepix, // input score
	const float* dL_depths, // input
	float* dL_dmean2D,
	float* dL_dconic,  // not in final output
	float* dL_dopacity,
	float* dL_dcolor,  // precompute color
	float* dL_dsemantic_feature, // semantic
	float* dL_dscore_feature, // score
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dz, // depth, not in final output
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;

	// block based semantic feature size
	float* collected_semantic_feature; 
	cudaMalloc((void**)&collected_semantic_feature, NUM_SEMANTIC_CHANNELS * BLOCK_SIZE * sizeof(float)); 

	// gaussian render image depth,
	const float* depth_ptr = geomState.depths;


	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,  // sorted gaussian list
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		semantic_feature,
		score_feature,
		depth_ptr, 
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,               // input
		dL_dfeaturepix, // input
		dL_dscorepix, // input
		dL_depths,                        // input
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor, // color
		dL_dsemantic_feature,  // semantic
		dL_dscore_feature, // score
		dL_dz, 				   // depth
		collected_semantic_feature
		), debug) 
		cudaFree(collected_semantic_feature);



	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot,
		dL_dz), debug)
}







