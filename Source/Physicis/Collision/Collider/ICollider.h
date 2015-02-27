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
		COLLIDER_INVALID, 
		COLLIDER_POINT, 
		COLLIDER_LINESEGMENT,
		COLLIDER_SPHERE,
		COLLIDER_BOX,
		COLLIDER_CONE, 
		COLLIDER_CAPSULE,
		COLLIDER_CYLINDER,
		COLLIDER_CONVEXHULL,
		COLLIDER_MAX_ID
	};

	ICollider(const vec3& center, Scene* scene);

	inline vec3 getPosition() const { return m_position; }
	inline void setPosition(const vec3& pos) { m_position = pos; }

	inline RigidBody* getRigidBody() const { return m_rigidBody; }
	void setRigidBody(RigidBody* rb) { m_rigidBody = rb; }

	uint getCollisionCount() { return m_collisionCount; }
	void resetCollisionCount() { m_collisionCount = 0; }
	void increaseCollisionCount() { ++m_collisionCount; }

	virtual QString className() { return "Collider"; }
	virtual void render(const float currentTime);

	void setMotionColor(const QColor& col);
	void setColor(const QColor& col);
	void resetColor();
	Scene* getScene() const { return m_scene; }

	inline const mat4& getTransformMatrix() { return m_transformMatrix; }

	ColliderType getColliderType() const { return m_colliderType; }
	void setCollisionObjectType(ColliderType type) { m_colliderType = type; }

	float getMargin() const { return m_margin; }
	void setMargin(float margin) { m_margin = margin; }

	/// Get the extreme vertex in the given direction
	virtual vec3 getLocalSupportPoint(const vec3& dir, float margin = 0) const = 0;

	/// Broad Phase Collision Detection
	virtual BroadPhaseCollisionFeedback onBroadPhase(ICollider* other) = 0;


protected:

	/*
	* For physics
	*/
	RigidBody* m_rigidBody;
	vec3 m_position;
	mat4 m_transformMatrix;
	float m_margin;
	ColliderType m_colliderType;
	mutable uint m_collisionCount;

	/*
	* For rendering
	*/
	virtual void init();
	virtual void drawElements(uint index);
	GLuint m_vao;
	GLuint m_vbo;
	ShadingTechniquePtr m_renderingEffect;
	QVector<MeshPtr> m_meshes;
	Scene* m_scene;
	QColor m_motionColor;
};

typedef QSharedPointer<ICollider> ColliderPtr;