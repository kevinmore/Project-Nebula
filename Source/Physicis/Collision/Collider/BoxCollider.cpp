#include "BoxCollider.h"
#include <Scene/Scene.h>
#include <Physicis/Entity/RigidBody.h>
using namespace Math;

BoxCollider::BoxCollider( const vec3& center, const vec3& halfExtents, Scene* scene )
	: ICollider(center, scene),
	  m_boxShape(center, halfExtents)
{
	m_colliderType = ICollider::COLLIDER_BOX;

	// the default model loaded here is a cube with half extent = 0.5, and the center is 0
	// we need to translate and scale it
	m_transformMatrix.translate(center);
	m_transformMatrix.scale(halfExtents.x() / 0.5f, halfExtents.y() / 0.5f, halfExtents.z() / 0.5f);

	// make the bounding box look slightly bigger than the actual one
	// this is for a better visual result
	m_transformMatrix.scale(1.02f);

	init();
}

BoxShape BoxCollider::getGeometryShape() const
{
	return m_boxShape;
}

void BoxCollider::init()
{
	ICollider::init();

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

BroadPhaseCollisionFeedback BoxCollider::onBroadPhase( ICollider* other )
{
	if (other->getColliderType() != ICollider::COLLIDER_BOX)
	{
		//qWarning() << "Collision detection between OBB and other colliders are not implemented yet.";
		return BroadPhaseCollisionFeedback();
	}
	BoxCollider* b = dynamic_cast<BoxCollider*>(other);
	
	// prepare feed back
	float distanceSquared = (getPosition() - b->getPosition()).lengthSquared();
	BroadPhaseCollisionFeedback notIntersecting(false, distanceSquared);
	BroadPhaseCollisionFeedback isIntersecting(true, distanceSquared);

	/// begin the separating axis tests
	float ra, rb;
	glm::mat3 R, AbsR;

	// get the size
	glm::vec3 ea = Converter::toGLMVec3(m_boxShape.getHalfExtents());
	glm::vec3 eb = Converter::toGLMVec3(b->getHalfExtents());

	glm::mat3 rotaionA = m_rigidBody->getRotationMatrix();
	glm::mat3 rotaionB = b->getRigidBody()->getRotationMatrix();

	// Compute rotation matrix expressing b in a¡¯s coordinate frame
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			R[i][j] = glm::dot(glm::row(rotaionA, i), glm::row(rotaionB, j));

	// Compute translation vector t
	glm::vec3 t = Converter::toGLMVec3(other->getPosition() - m_position);
	// Bring translation into a¡¯s coordinate frame
	float tx = glm::dot(t, glm::row(rotaionA, 0));
	float ty = glm::dot(t, glm::row(rotaionA, 1));
	float tz = glm::dot(t, glm::row(rotaionA, 2));
	t = glm::vec3(tx, ty, tz);

	// Compute common subexpressions. Add in an epsilon term to
	// counteract arithmetic errors when two edges are parallel and
	// their cross product is (near) null
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			AbsR[i][j] = glm::abs(R[i][j]) + 0.001;

	// Test axes L = A0, L = A1, L = A2
	for (int i = 0; i < 3; ++i) {
		ra = ea[i];
		rb = eb[0] * AbsR[i][0] + eb[1] * AbsR[i][1] + eb[2] * AbsR[i][2];
		if (glm::abs(t[i]) > ra + rb)
			return notIntersecting;
	}

	// Test axes L = B0, L = B1, L = B2
	for (int i = 0; i < 3; ++i) {
		ra = ea[0] * AbsR[0][i] + ea[1] * AbsR[1][i] + ea[2] * AbsR[2][i];
		rb = eb[i];
		if (glm::abs(t[0] * R[0][i] + t[1] * R[1][i] + t[2] * R[2][i]) > ra + rb)
			return notIntersecting;
	}

	// Test axis L = A0 x B0
	ra = ea[1] * AbsR[2][0] + ea[2] * AbsR[1][0];
	rb = eb[1] * AbsR[0][2] + eb[2] * AbsR[0][1];
	if (glm::abs(t[2] * R[1][0] - t[1] * R[2][0]) > ra + rb) 
		return notIntersecting;

	// Test axis L = A0 x B1
	ra = ea[1] * AbsR[2][1] + ea[2] * AbsR[1][1];
	rb = eb[0] * AbsR[0][2] + eb[2] * AbsR[0][0];
	if (glm::abs(t[2] * R[1][1] - t[1] * R[2][1]) > ra + rb)
		return notIntersecting;

	// Test axis L = A0 x B2
	ra = ea[1] * AbsR[2][2] + ea[2] * AbsR[1][2];
	rb = eb[0] * AbsR[0][1] + eb[1] * AbsR[0][0];
	if (glm::abs(t[2] * R[1][2] - t[1] * R[2][2]) > ra + rb)
		return notIntersecting;

	// Test axis L = A1 x B0
	ra = ea[0] * AbsR[2][0] + ea[2] * AbsR[0][0];
	rb = eb[1] * AbsR[1][2] + eb[2] * AbsR[1][1];
	if (glm::abs(t[0] * R[2][0] - t[2] * R[0][0]) > ra + rb)
		return notIntersecting;

	// Test axis L = A1 x B1
	ra = ea[0] * AbsR[2][1] + ea[2] * AbsR[0][1];
	rb = eb[0] * AbsR[1][2] + eb[2] * AbsR[1][0];
	if (glm::abs(t[0] * R[2][1] - t[2] * R[0][1]) > ra + rb)
		return notIntersecting;

	// Test axis L = A1 x B2
	ra = ea[0] * AbsR[2][2] + ea[2] * AbsR[0][2];
	rb = eb[0] * AbsR[1][1] + eb[1] * AbsR[1][0];
	if (glm::abs(t[0] * R[2][2] - t[2] * R[0][2]) > ra + rb)
		return notIntersecting;

	// Test axis L = A2 x B0
	ra = ea[0] * AbsR[1][0] + ea[1] * AbsR[0][0];
	rb = eb[1] * AbsR[2][2] + eb[2] * AbsR[2][1];
	if (glm::abs(t[1] * R[0][0] - t[0] * R[1][0]) > ra + rb)
		return notIntersecting;

	// Test axis L = A2 x B1
	ra = ea[0] * AbsR[1][1] + ea[1] * AbsR[0][1];
	rb = eb[0] * AbsR[2][2] + eb[2] * AbsR[2][0];
	if (glm::abs(t[1] * R[0][1] - t[0] * R[1][1]) > ra + rb)
		return notIntersecting;

	// Test axis L = A2 x B2
	ra = ea[0] * AbsR[1][2] + ea[1] * AbsR[0][2];
	rb = eb[0] * AbsR[2][1] + eb[1] * AbsR[2][0];
	if (glm::abs(t[1] * R[0][2] - t[0] * R[1][2]) > ra + rb)
		return notIntersecting;

	// Since no separating axis is found, the OBBs must be intersecting
	return isIntersecting;
}

void BoxCollider::setHalfExtents( const vec3& halfExtents )
{
	// scale the shape compared to the current halfExtents
	vec3 oldHalfExtents = m_boxShape.getHalfExtents();
	vec3 scale(halfExtents.x() / oldHalfExtents.x(), halfExtents.y() / oldHalfExtents.y(), halfExtents.z() / oldHalfExtents.z());
	m_transformMatrix.scale(scale);
	
	m_boxShape.setHalfExtents(halfExtents);

	// re-compute the inertia tensor
	if (m_rigidBody)
	{
		m_rigidBody->computeInertiaTensor();
	}
}

vec3 BoxCollider::getHalfExtents() const
{
	return m_boxShape.getHalfExtents();
}

vec3 BoxCollider::getLocalSupportPoint( const vec3& dir, float margin /*= 0*/ ) const
{

	vec3 halfExtents = m_boxShape.getHalfExtents();

	vec3 supportPoint;

	supportPoint.setX(dir.x() < 0 ? -halfExtents.x() - margin : halfExtents.x() + margin);
	supportPoint.setY(dir.y() < 0 ? -halfExtents.y() - margin : halfExtents.y() + margin);
	supportPoint.setZ(dir.z() < 0 ? -halfExtents.z() - margin : halfExtents.z() + margin);

	return supportPoint;
}

void BoxCollider::setScale( const vec3& scale )
{
	// reset the transform matrix
	m_transformMatrix.setToIdentity();

	// change the half extents and the center
	vec3 center = m_boxShape.getCenter();
	m_boxShape.setCenter(vec3(center.x() * scale.x(), 
							  center.y() * scale.y(), 
							  center.z() * scale.z()));

	m_transformMatrix.translate(m_boxShape.getCenter());
	
	vec3 halfExtents = m_boxShape.getHalfExtents();
	// temporary set the half extents to default
	// because this function (setScale()) is only called when the scale of the game object is changed
	// in this way, we set the transform matrix to identity, and the size of the box to default (0.5)
	m_boxShape.setHalfExtents(vec3(0.5f, 0.5f, 0.5f));
	setHalfExtents(vec3(halfExtents.x() * scale.x(), 
						halfExtents.y() * scale.y(), 
						halfExtents.z() * scale.z()));

	// finally scale the transform matrix a bit larger to make sure it can be seen
	m_transformMatrix.scale(1.02f);
}

vec3 BoxCollider::getAABBMinLocal() const
{
	return m_boxShape.getCenter() - m_boxShape.getHalfExtents();
}

vec3 BoxCollider::getAABBMaxLocal() const
{
	return m_boxShape.getCenter() + m_boxShape.getHalfExtents();
}

