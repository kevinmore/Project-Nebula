#pragma once
#include "IRender.h"
#include "ShadingTechniques/ShadowMapFBO.h"

class RenderGLSL :	public IRender
{
public:
	static RenderGLSL* instance();

	bool startup();
	void shutdown();

	void draw();
	void toggleShadows();
	void drawShadows();

protected:
	void setTransform(const mat4 &projection, const mat4 &modelview, bool infPersp);

	//// GI
	enum {giDim = 256};
	enum {giAxisCount = 6};
	unsigned int _giTex[giAxisCount];

	unsigned int _giFramebuffer;

	float _voxelScale;
	vec3 _voxelBias;

	void initGI();
	void shutdownGI();
	bool startupGI();
	void updateGI();
	void updateGI_drawScene(const mat4 &mvp, const mat4 &storageTransform);
	void updateGI_mipmap(int direction);
	////

	//// Shadows
	bool _drawShadows;
	mat4 _shadowMVP;
	ShadowMapFBO m_shadowMapFBO;

	void shutdownShadows();
	bool startupShadows();
	////


	RenderOptions _renderOptions;
	mat4 _projection, _modelview, _modelviewInverse, _mvp, _mvpInverse;

private:
	RenderGLSL();
	~RenderGLSL();
	static RenderGLSL* m_instance;
};

