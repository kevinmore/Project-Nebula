#include "RenderGLSL.h"
#include <Scene/Scene.h>

RenderGLSL::RenderGLSL()
	: IRender()
{
	initGI();
	initShadows();
	shutdown();

	m_scene = Scene::instance();
}

RenderGLSL::~RenderGLSL()
{
	shutdown();
}

void RenderGLSL::shutdown()
{
	shutdownGI();
	shutdownShadows();
}

bool RenderGLSL::startup()
{
	shutdown();

#if defined(ENABLE_DEBUG_OUTPUT_CALLBACK)
	if (GL_TRUE == glewIsSupported("GL_KHR_debug"))
	{
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(DebugOutputCallback, 0);
		glEnable(GL_DEBUG_OUTPUT);
	}
	else if (GL_TRUE == glewIsSupported("GL_ARB_debug_output"))
	{
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
		glDebugMessageCallbackARB(DebugOutputCallback, 0);
	}
#endif

	// enable sRGB backbuffer
	// RGB8 and RGBA8 backbuffers will likely be sRGB-capable, and this enable is all that is needed to turn on sRGB for
	// the default framebuffer
	// for FBOs, the internal format of the color attachment(s) needs to be an sRGB format for this to have any effect
	glEnable(GL_FRAMEBUFFER_SRGB);

	if (!startupGI())
	{
		qWarning() << "RenderGLSL::startupGI failed";
		return false;
	}

	if (!startupShadows())
	{
		qWarning() << "RenderGLSL::startupShadows failed";
		return false;
	}

	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	return true;
}

void RenderGLSL::draw( float minFPS, float averageFPS, float maxFPS, float t )
{
	// draw shadows before setting up the main render target (it would overwrite the main RT settings otherwise)
	drawShadows();
}

void RenderGLSL::toggleShadows()
{
	_drawShadows = !_drawShadows;
}

void RenderGLSL::initGI()
{
	for (int n = 0; n < giAxisCount; n++)
	{
		_giTex[n] = 0;
		_giTexDXT[n] = 0;
	}

	_giFramebuffer = 0;

	for (int n = 0; n < giDXTBufferCount; n++)
	{
		_giDXTBuffer[n] = 0;
	}

	_giDebugPointBuffer = 0;
}

void RenderGLSL::shutdownGI()
{
	for (int n = 0; n < giAxisCount; n++)
	{
		glDeleteTextures(1, &_giTex[n]);
		_giTex[n] = 0;

		glDeleteTextures(1, &_giTexDXT[n]);
		_giTexDXT[n] = 0;
	}

	if (_giFramebuffer)
		glDeleteFramebuffers(1, &_giFramebuffer);
	_giFramebuffer = 0;

	for (int n = 0; n < giDXTBufferCount; n++)
	{
		if (_giDXTBuffer[n])
			glDeleteBuffers(1, &_giDXTBuffer[n]);
		_giDXTBuffer[n] = 0;
	}

	if (_giDebugPointBuffer)
		glDeleteBuffers(1, &_giDebugPointBuffer);
	_giDebugPointBuffer = 0;
}

bool RenderGLSL::startupGI()
{
	_voxelScale = 0.0003f;
	_voxelBias = vec3(1920, 126, 1182);

	for (int n = 0; n < giAxisCount; n++)
	{
		glGenTextures(1, &_giTex[n]);
		glBindTexture(GL_TEXTURE_3D, _giTex[n]);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		Math::Random::random();
		int mipLevels = Math::Util::getBitIndex(giDim) + 1;
		for (int mip = 0; mip < mipLevels; mip++)
		{
			glTexImage3D(GL_TEXTURE_3D, mip, GL_RGBA8, giDim >> mip, giDim >> mip, giDim >> mip, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		}


		glGenTextures(1, &_giTexDXT[n]);
		glBindTexture(GL_TEXTURE_3D, _giTexDXT[n]);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

		int compressedBlockSize = 16; // DXT3/5
		for (int mip = 0; mip < mipLevels; mip++)
		{
			int dim = giDim >> mip;
			int size = ((dim + 3) / 4) * ((dim + 3) / 4) * compressedBlockSize * dim;

			glCompressedTexImage3D(GL_TEXTURE_3D, mip, GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, dim, dim, dim, 0, size, nullptr);
		}
	}

	glGenFramebuffers(1, &_giFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, _giFramebuffer);
	glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, giDim);
	glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, giDim);
	glFramebufferParameteri(GL_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_LAYERS, giAxisCount);

	_giDXTBufferIndex = 0;
	for (int n = 0; n < giDXTBufferCount; n++)
	{
		glGenBuffers(1, &_giDXTBuffer[n]);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, _giDXTBuffer[n]);
		glBufferData(GL_SHADER_STORAGE_BUFFER, giDXTBufferSize, nullptr, GL_STATIC_DRAW);
	}

	// 	{
	// 		glGenBuffers(1, &_giDebugPointBuffer);
	// 		glBindBuffer(GL_ARRAY_BUFFER, _giDebugPointBuffer);
	// 		glBufferData(GL_ARRAY_BUFFER, giDim * giDim * giDim * sizeof(float) * 6, nullptr, GL_STATIC_DRAW);
	// 		float *debugPointData = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	// 
	// 		float step = 1.0f / giDim;
	// 		float invVoxelScale = 1.0f / _voxelScale;
	// 		for (float z = .5f * step; z < 1.0f; z += step)
	// 		{
	// 			for (float y = .5f * step; y < 1.0f; y += step)
	// 			{
	// 				for (float x = .5f * step; x < 1.0f; x += step)
	// 				{
	// 					vector3 texPos(x, y, z);
	// 					vector3 worldPos = texPos * invVoxelScale - _voxelBias;
	// 
	// 					debugPointData[0] = worldPos.x;
	// 					debugPointData[1] = worldPos.y;
	// 					debugPointData[2] = worldPos.z;
	// 
	// 					debugPointData[3] = texPos.x;
	// 					debugPointData[4] = texPos.y;
	// 					debugPointData[5] = texPos.z;
	// 
	// 					debugPointData += 6;
	// 				}
	// 			}
	// 		}
	// 
	// 		glUnmapBuffer(GL_ARRAY_BUFFER);
	// 	}

	return true;
}

void RenderGLSL::initShadows()
{
	_shadowTex = 0;
	_shadowFramebuffer = 0;
}

void RenderGLSL::shutdownShadows()
{
	if (_shadowTex)
		glDeleteTextures(1, &_shadowTex);
	_shadowTex = 0;

	if (_shadowFramebuffer)
		glDeleteFramebuffers(1, &_shadowFramebuffer);
	_shadowFramebuffer = 0;
}

bool RenderGLSL::startupShadows()
{
	glGenTextures(1, &_shadowTex);
	glBindTexture(GL_TEXTURE_2D, _shadowTex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, _renderOptions.shadowMapResolution, _renderOptions.shadowMapResolution, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, 0);

	glGenFramebuffers(1, &_shadowFramebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, _shadowFramebuffer);
	glReadBuffer(GL_NONE);
	glDrawBuffer(GL_NONE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, _shadowTex, 0);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return true;
}

void RenderGLSL::drawShadows()
{
	vec3 cameraPos = m_scene->getCamera()->position();
	float shadowRadius = 1.0f / _voxelScale; // world space radius of shadow buffer
	vec3 lightDir = m_scene->getLights().first()->direction();

	glBindFramebuffer(GL_FRAMEBUFFER, _shadowFramebuffer);
	glColorMask(0, 0, 0, 0);

	glClearDepth(1);
	glClear(GL_DEPTH_BUFFER_BIT);

	mat4 shadowProjection;
	shadowProjection.ortho(-shadowRadius, shadowRadius, -shadowRadius, shadowRadius, -shadowRadius, shadowRadius);

	// for this demo, just leave the shadows fixed - the world is small enough that the shadow buffer doesn't need to follow the camera around
	mat4 shadowModelview;
	shadowModelview.lookAt(lightDir, vec3(0, 0, 0), vec3(0, 1, 0));// * matrix().translate(-cameraPos);
	_shadowMVP = _mvp;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glColorMask(1, 1, 1, 1);
}

void RenderGLSL::setTransform( const mat4 &projection, const mat4 &modelview, bool infPersp )
{
	_projection = projection;

	_modelview = modelview;
	_modelviewInverse = modelview.inverted();

	_mvp = _projection * _modelview;
	_mvpInverse = _mvp.inverted();
}


/*
Update overview:
For this demo, the entire scene is regenerated into voxel form each frame. Caching or generating portions of the scene
at a time (to reduce memory requirements and per-frame rebuild cost) are possible extensions.

Steps:
	-clear and reset
	-generate all touched voxels for all primitives
		-render scene from 6 directions (+X,-X,+Y,-Y,+Z,-Z)
		-(alternative: render geometry only once, selecting major axis, use conservative rasterization in GS, use conservative depth in FS
		to ensure all touched depth layers get a voxel)
	-light and shade voxels
		-(combined with generation step in this demo... could be deferred to a CS pass, reducing duplicate shading, like in Unreal 4)
	-dispatch voxels to 6 volume textures, one for each direction
		-TODO: when multiple voxels touch the same location, the winner is determined by which one is 'nearest' as determined by the texture's direction
	-generate mipmaps
		-for each direction texture, determine the color along each column in the source mip 2x2x2 (column meaning looking down the direction of
		the texture)
			-use standard alpha blending to determine color and opacity
		-output texel is average of each of the four columns
	-(optional) DXT volume texture compression
*/
void RenderGLSL::updateGI()
{
	glBindFramebuffer(GL_FRAMEBUFFER, _giFramebuffer);
	glViewport(0, 0, giDim, giDim);

	// clear
	glClearColor(0, 0, 0, 0);
	for (int n = 0; n < giAxisCount; n++)
	{
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _giTex[n], 0);
		glClear(GL_COLOR_BUFFER_BIT);
	}
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 0, 0);

	// generate voxels
	for (int n = 0; n < giAxisCount; n++)
	{
// 		matrix worldToUnitCube = 
// 			matrix().translate(-1.0f, -1.0f, -1.0f) * matrix().scale(2.0f, 2.0f, 2.0f, 1.0f) *
// 			matrix().scale(_voxelScale, _voxelScale, _voxelScale, 1.0f) * matrix().translate(_voxelBias);
		mat4 worldToUnitCube;
		worldToUnitCube.translate(_voxelBias);
		worldToUnitCube.scale(_voxelScale);
		worldToUnitCube.translate(-1.0f, -1.0f, -1.0f);

		mat4 unitCubeToNDC = Math::Matrix4::cubemapMatrix(GL_TEXTURE_CUBE_MAP_POSITIVE_X + n);
		
		mat4 worldToNDC = unitCubeToNDC * worldToUnitCube;

		mat4 ndcToTex = unitCubeToNDC.transposed();

		glBindImageTexture(0, _giTex[n], 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA8);

		updateGI_drawScene(worldToNDC, ndcToTex);
	}

	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	for (int n = 0; n < giAxisCount; n++)
	{
		updateGI_mipmap(n);
	}

	glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void RenderGLSL::updateGI_drawScene( const mat4 &mvp, const mat4 &storageTransform )
{

}

void RenderGLSL::updateGI_mipmap( int direction )
{

}
