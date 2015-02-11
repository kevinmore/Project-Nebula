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

	inline vec3 getCenter() const { return m_center; }
	inline void setCenter(const vec3& pos) { m_center = pos; }

	inline RigidBody* getRigidBody() const { return m_rigidBody; }
	void setRigidBody(RigidBody* rb) { m_rigidBody = rb; }

	virtual BroadPhaseCollisionFeedback onBroadPhase(ICollider* other) = 0;

	virtual QString className() { return "Collider"; }
	virtual void render(const float currentTime);

	void setColor(const QColor& col);
	Scene* getScene() const { return m_scene; }

	inline const mat4& getTransformMatrix() { return m_transformMatrix; }

	ColliderType getColliderType() const { return m_colliderType; }
	void setCollisionObjectType(ColliderType type) { m_colliderType = type; }

	float getMargin() const { return m_margin; }
	void setMargin(float margin) { m_margin = margin; }

	void setSize(float x, float y, float z) { m_halfExtent = vec3(x/2.0f, y/2.0f, z/2.0f); }
	vec3 getSize() const { return 2.0 * m_halfExtent; }

	void setVertices(const QVector<vec3> vertices) { m_vertices = vertices; }
	QVector<vec3> getVertices() const { return m_vertices; }

	/// Get the extreme vertex in the given direction
	vec3 getLocalSupportPoint(const vec3& dir, float margin = 0) const;

protected:

	/*
	* For physics
	*/
	RigidBody* m_rigidBody;
	vec3 m_center;
	mat4 m_transformMatrix;
	float m_margin;
	ColliderType m_colliderType;
	vec3 m_halfExtent;
	QVector<vec3> m_vertices;

	/*
	* For rendering
	*/
	virtual void init();
	void drawElements(uint index);
	GLuint m_vao;
	ShadingTechniquePtr m_renderingEffect;
	QVector<MeshPtr> m_meshes;
	Scene* m_scene;
};

typedef QSharedPointer<ICollider> ColliderPtr;