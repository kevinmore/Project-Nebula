#pragma once
#include "CollisionFeedback.h"
#include <Utility/Math.h>
#include <Primitives/Component.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>
#include <Scene/Managers/MeshManager.h>
#include <QOpenGLFunctions_4_3_Core>

class Scene;
class AbstractCollider : public Component, protected QOpenGLFunctions_4_3_Core
{
public:
	AbstractCollider(const vec3& center, Scene* scene);

	inline vec3 getCenter() const { return m_center; }

	virtual CollisionFeedback intersect(AbstractCollider* other) = 0;

	virtual QString className() { return "Collider"; }
	virtual void render(const float currentTime);

protected:
	virtual void init();
	void drawElements(uint index);
	GLuint m_vao;
	ShadingTechniquePtr m_renderingEffect;
	QVector<MeshPtr> m_meshes;
	mat4 m_transformMatrix;

	Scene* m_scene;
	vec3 m_center;
};

typedef QSharedPointer<AbstractCollider> ColliderPtr;