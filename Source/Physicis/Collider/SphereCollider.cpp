#include "SphereCollider.h"
#include <Scene/Scene.h>


SphereCollider::SphereCollider( const vec3& center, const float radius, Scene* scene )
	: AbstractCollider(center, scene)
{
	m_sphereShape = SphereShape(center, radius);
	m_colliderType = AbstractCollider::COLLIDER_SPHERE;

	// the default model loaded here is a sphere with radius = 0.5, and the center is 0
	// we need to translate and scale it
	m_transformMatrix.translate(m_center);
	m_transformMatrix.scale(radius / 0.5f);

	init();
}

SphereShape SphereCollider::getGeometryShape() const
{
	return m_sphereShape;
}

CollisionFeedback SphereCollider::intersect( AbstractCollider* other )
{
	if (other->m_colliderType != AbstractCollider::COLLIDER_SPHERE)
	{
		//qWarning() << "Collision detection between sphere and other colliders are not implemented yet.";
		return CollisionFeedback();
	}
	SphereCollider* sp = dynamic_cast<SphereCollider*>(other);
	float radiusSum = m_sphereShape.getRadius() + sp->getGeometryShape().getRadius();
	float centerDisSqaure = (m_center - sp->getCenter()).lengthSquared();

	return CollisionFeedback(centerDisSqaure < radiusSum, centerDisSqaure - radiusSum * radiusSum);
}

void SphereCollider::init()
{
	AbstractCollider::init();

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

void SphereCollider::setRadius( const float radius )
{
	m_sphereShape.setRadius(radius);

	// resize the transform matrix
	m_transformMatrix.setToIdentity();
	m_transformMatrix.scale(radius / 0.5f);
}
