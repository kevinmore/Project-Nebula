#include "BoxCollider.h"
#include <Scene/Scene.h>

BoxCollider::BoxCollider( const vec3& center, const vec3& halfExtents, Scene* scene )
	: AbstractCollider(center, scene)
{
	m_boxShape = BoxShape(center, halfExtents);

	// the default model loaded here is a cube with half extent = 0.5
	// we need to scale it
	m_transformMatrix.scale(halfExtents.x() / 0.5f, halfExtents.y() / 0.5f, halfExtents.z() / 0.5f);
	init();
}

BoxShape BoxCollider::getGeometryShape() const
{
	return m_boxShape;
}

CollisionFeedback BoxCollider::intersect( AbstractCollider* other )
{
	return CollisionFeedback();
}

void BoxCollider::init()
{
	AbstractCollider::init();

	ModelLoader loader;
	QVector<ModelDataPtr> modelDataVector = loader.loadModel("../Resource/Models/Common/cube.obj", m_renderingEffect->getShaderProgram()->programId());
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

void BoxCollider::setHalfExtents( const vec3& halfExtents )
{
	m_boxShape.setHalfExtents(halfExtents);

	// resize the transform matrix
	m_transformMatrix.setToIdentity();
	m_transformMatrix.scale(halfExtents.x() / 0.5f, halfExtents.y() / 0.5f, halfExtents.z() / 0.5f);
}
