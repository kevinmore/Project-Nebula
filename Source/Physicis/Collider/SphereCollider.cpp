#include "SphereCollider.h"
#include <Scene/Scene.h>


SphereCollider::SphereCollider( const vec3& center, const float radius, Scene* scene )
	: AbstractCollider(center, scene)
{
	m_sphereShape = SphereShape(center, radius);
	init();
}

SphereShape SphereCollider::getGeometryShape() const
{
	return m_sphereShape;
}

CollisionFeedback SphereCollider::intersect( AbstractCollider* other )
{
	SphereCollider* sp = dynamic_cast<SphereCollider*>(other);
	float radiusSum = m_sphereShape.getRadius() + sp->getGeometryShape().getRadius();
	float centerDis = (m_center - sp->getCenter()).length();

	return CollisionFeedback(centerDis > radiusSum, centerDis - radiusSum);
}

void SphereCollider::init()
{
	AbstractCollider::init();

	ModelLoader loader;
	QVector<ModelDataPtr> modelDataVector = loader.loadModel("../Resource/Models/Common/sphere.obj", m_renderingEffect->getShaderProgram()->programId());
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
