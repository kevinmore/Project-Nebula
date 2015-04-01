#pragma once
#include "IRender.h"

class RenderGLSL :	public IRender
{
public:
	RenderGLSL();
	~RenderGLSL();

	bool startup();
	void shutdown();

	void draw(float minFPS, float averageFPS, float maxFPS, float t);
	void toggleShadows();

protected:
	void setTransform(const mat4 &projection, const mat4 &modelview, bool infPersp);

	//// GI
	enum {giDim = 256};
	enum {giAxisCount = 6};
	unsigned int _giTex[giAxisCount];
	unsigned int _giTexDXT[giAxisCount];

	unsigned int _giFramebuffer;

	enum {giDXTBufferSize = giDim * giDim * giDim}; // DXT3/5, 1 byte per pixel
	enum {giDXTBufferCount = 1};
	unsigned int _giDXTBuffer[giDXTBufferCount];
	int _giDXTBufferIndex;

	unsigned int _giDebugPointBuffer;

	float _voxelScale;
	vec3 _voxelBias;

	void initGI();
	void shutdownGI();
	bool startupGI();
	void updateGI();
	void updateGI_drawScene(const mat4 &mvp, const mat4 &storageTransform);
	void updateGI_mipmap(int direction);
	void updateGI_dxt(int direction);

	void drawDebugGI();
	////

	//// Shadows
	unsigned int _shadowTex;
	unsigned int _shadowFramebuffer;
	bool _drawShadows;
	mat4 _shadowMVP;

	void initShadows();
	void shutdownShadows();
	bool startupShadows();
	void drawShadows();
	////


	RenderOptions _renderOptions;
	mat4 _projection, _modelview, _modelviewInverse, _mvp, _mvpInverse;
};

