#pragma once
struct RenderOptions
{
	float drawDist;

	int maxDynamicLights;

	// 0 disables shadow maps
	int shadowMapCascadeCount;
	int shadowMapResolution;

	bool parallaxMapping;
	bool reliefMapping;
	bool tessellation;

	bool textureArrays;

	bool vsync;

	RenderOptions()
		: drawDist(100.0f)
		, maxDynamicLights(0)
		, shadowMapCascadeCount(4)
		, shadowMapResolution(4096)
		, parallaxMapping(true)
		, reliefMapping(true)
		, tessellation(true)
		, textureArrays(true)
		, vsync(false)
	{}
};