#pragma once
#include "RenderOptions.h"
#include <Utility/EngineCommon.h>
#include <QOpenGLFunctions_4_3_Core>

class Scene;
class IRender : protected QOpenGLFunctions_4_3_Core
{
public:
	IRender() { Q_ASSERT(initializeOpenGLFunctions()); }
	virtual ~IRender() {}

	virtual bool startup() = 0;
	virtual void shutdown() = 0;

	virtual void draw() = 0;

	virtual void toggleShadows() = 0;
	//virtual void adjustLightDir(const vec3& adjustment) = 0;

protected:

	Scene* m_scene;
};