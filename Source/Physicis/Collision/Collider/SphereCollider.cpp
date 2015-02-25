#include "SphereCollider.h"
#include <Scene/Scene.h>


SphereCollider::SphereCollider( const vec3& center, const float radius, Scene* scene )
	: ICollider(center, scene),
	  m_sphereShape(center, radius)
{
	m_colliderType = ICollider::COLLIDER_SPHERE;

	// the default model loaded here is a sphere with radius = 0.5, and the center is 0
	// we need to translate and scale it
	m_transformMatrix.translate(m_center);
	m_transformMatrix.scale(radius / 0.5f);

	// make the bounding box look slightly bigger than the actual one
	// this is for a better visual result
	m_transformMatrix.scale(1.05f);

	init();
}

// SphereShape SphereCollider::getGeometryShape() const
// {
// 	return m_sphereShape;
// }

BroadPhaseCollisionFeedback SphereCollider::onBroadPhase( ICollider* other )
{
	if (other->getColliderType() != ICollider::COLLIDER_SPHERE)
	{
		//qWarning() << "Collision detection between sphere and other colliders are not implemented yet.";
		return BroadPhaseCollisionFeedback();
	}
	SphereCollider* sp = dynamic_cast<SphereCollider*>(other);
	float radiusSum = m_sphereShape.getRadius() + sp->getRadius();
	float centerDisSqaure = (m_center - sp->getCenter()).lengthSquared();

	return BroadPhaseCollisionFeedback(centerDisSqaure < radiusSum, centerDisSqaure - radiusSum * radiusSum);
}

void SphereCollider::init()
{
	ICollider::init();

	ModelLoader loader;
	QVector<ModelDataPtr> modelDataVector = loader.loadModel("../Resource/Models/Common/boundingsphere.obj", m_renderingEffect->getShaderProgram()->programId());
	m_vao = loader.getVAO();

	// traverse modelData vector
	for (int i = 0; i < modelDataVector.size(); ++i)
	{
		ModelDataPtr data = modelDataVector[i];
		// deal with the mesh
		MeshPtr mesh(new Mesh(data->meshData.name, data->meshData.numIndices, data->meshData.baseVertex, data->meshData.baseIndex));
		m_meshes.push_back(mesh);
	}
}

float SphereCollider::getRadius() const
{
	return m_sphereShape.getRadius();
}

void SphereCollider::setRadius( const float radius )
{
	m_sphereShape.setRadius(radius);
}

vec3 SphereCollider::getLocalSupportPoint( const vec3& dir, float margin /*= 0*/ ) const
{
	float radius = m_sphereShape.getRadius();
	return (radius + margin) * dir.normalized();
}
