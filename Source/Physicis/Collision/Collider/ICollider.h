#pragma once
#include <Physicis/Collision/BroadPhase/BroadPhaseCollisionFeedback.h>
#include <Utility/Math.h>
#include <Primitives/Component.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>
#include <Scene/Managers/MeshManager.h>
#include <QOpenGLFunctions_4_3_Core>

class Scene;
class RigidBody;
class ICollider : public Component, protected QOpenGLFunctions_4_3_Core
{
public:
	
	enum ColliderType
	{
		COLLIDER_SPHERE,
		COLLIDER_BOX,
		COLLIDER_MAX_ID
	};

	ICollider(const vec3& center, Scene* scene);

	inline vec3 getCenter() const { return m_center; }
	inline void setCenter(const vec3& pos) { m_center = pos; }

	inline RigidBody* getRigidBody() const { return m_rigidBody; }
	void setRigidBody(RigidBody* rb) { m_rigidBody = rb; }

	virtual BroadPhaseCollisionFeedback intersect(ICollider* other) = 0;

	virtual QString className() { return "Collider"; }
	virtual void render(const float currentTime);

	void setColor(const QColor& col);
	Scene* getScene() const { return m_scene; }

	inline const mat4& getTransformMatrix() { return m_transformMatrix; }

	ColliderType m_colliderType;

protected:
	virtual void init();
	void drawElements(uint index);
	GLuint m_vao;
	ShadingTechniquePtr m_renderingEffect;
	QVector<MeshPtr> m_meshes;
	mat4 m_transformMatrix;
	RigidBody* m_rigidBody;
	Scene* m_scene;
	vec3 m_center;
};

typedef QSharedPointer<ICollider> ColliderPtr;