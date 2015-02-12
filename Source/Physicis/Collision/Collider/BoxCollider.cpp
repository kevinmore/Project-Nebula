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
	m_transformMatrix.translate(m_center);
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

BroadPhaseCollisionFeedback BoxCollider::onBroadPhase( ICollider* other )
{
	if (other->getColliderType() != ICollider::COLLIDER_BOX)
	{
		//qWarning() << "Collision detection between OBB and other colliders are not implemented yet.";
		return BroadPhaseCollisionFeedback();
	}
	BoxCollider* b = dynamic_cast<BoxCollider*>(other);
	
	// prepare feed back
	float distanceSquared = (getCenter() - b->getCenter()).lengthSquared();
	BroadPhaseCollisionFeedback notIntersecting(false, distanceSquared);
	BroadPhaseCollisionFeedback isIntersecting(true, distanceSquared);

	/// begin the separating axis tests
	float ra, rb;
	mat3 R, AbsR;

	// get the size
	vec3 ea = m_boxShape.getHalfExtents();
	vec3 eb = b->getGeometryShape().getHalfExtents();

	mat3 rotaionA = m_rigidBody->getRotationMatrix();
	mat3 rotaionB = b->getRigidBody()->getRotationMatrix();


	// Compute rotation matrix expressing b in a¡¯s coordinate frame
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			R.m[i][j] = vec3::dotProduct(Matrix3::getRow(rotaionA, i), Matrix3::getRow(rotaionB, j));

	// Compute translation vector t
	vec3 t = other->getCenter() - m_center;
	// Bring translation into a¡¯s coordinate frame
	float tx = vec3::dotProduct(t, Matrix3::getRow(rotaionA, 0));
	float ty = vec3::dotProduct(t, Matrix3::getRow(rotaionA, 1));
	float tz = vec3::dotProduct(t, Matrix3::getRow(rotaionA, 2));
	t = vec3(tx, ty, tz);

	// Compute common subexpressions. Add in an epsilon term to
	// counteract arithmetic errors when two edges are parallel and
	// their cross product is (near) null
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			AbsR.m[i][j] = qAbs(R.m[i][j]) + 0.001;

	// Test axes L = A0, L = A1, L = A2
	for (int i = 0; i < 3; i++) {
		ra = ea[i];
		rb = eb[0] * AbsR.m[i][0] + eb[1] * AbsR.m[i][1] + eb[2] * AbsR.m[i][2];
		if (qAbs(t[i]) > ra + rb)
			return notIntersecting;
	}

	// Test axes L = B0, L = B1, L = B2
	for (int i = 0; i < 3; i++) {
		ra = ea[0] * AbsR.m[0][i] + ea[1] * AbsR.m[1][i] + ea[2] * AbsR.m[2][i];
		rb = eb[i];
		if (qAbs(t[0] * R.m[0][i] + t[1] * R.m[1][i] + t[2] * R.m[2][i]) > ra + rb)
			return notIntersecting;
	}

	// Test axis L = A0 x B0
	ra = ea[1] * AbsR.m[2][0] + ea[2] * AbsR.m[1][0];
	rb = eb[1] * AbsR.m[0][2] + eb[2] * AbsR.m[0][1];
	if (qAbs(t[2] * R.m[1][0] - t[1] * R.m[2][0]) > ra + rb) 
		return notIntersecting;

	// Test axis L = A0 x B1
	ra = ea[1] * AbsR.m[2][1] + ea[2] * AbsR.m[1][1];
	rb = eb[0] * AbsR.m[0][2] + eb[2] * AbsR.m[0][0];
	if (qAbs(t[2] * R.m[1][1] - t[1] * R.m[2][1]) > ra + rb)
		return notIntersecting;

	// Test axis L = A0 x B2
	ra = ea[1] * AbsR.m[2][2] + ea[2] * AbsR.m[1][2];
	rb = eb[0] * AbsR.m[0][1] + eb[1] * AbsR.m[0][0];
	if (qAbs(t[2] * R.m[1][2] - t[1] * R.m[2][2]) > ra + rb)
		return notIntersecting;

	// Test axis L = A1 x B0
	ra = ea[0] * AbsR.m[2][0] + ea[2] * AbsR.m[0][0];
	rb = eb[1] * AbsR.m[1][2] + eb[2] * AbsR.m[1][1];
	if (qAbs(t[0] * R.m[2][0] - t[2] * R.m[0][0]) > ra + rb)
		return notIntersecting;

	// Test axis L = A1 x B1
	ra = ea[0] * AbsR.m[2][1] + ea[2] * AbsR.m[0][1];
	rb = eb[0] * AbsR.m[1][2] + eb[2] * AbsR.m[1][0];
	if (qAbs(t[0] * R.m[2][1] - t[2] * R.m[0][1]) > ra + rb)
		return notIntersecting;

	// Test axis L = A1 x B2
	ra = ea[0] * AbsR.m[2][2] + ea[2] * AbsR.m[0][2];
	rb = eb[0] * AbsR.m[1][1] + eb[1] * AbsR.m[1][0];
	if (qAbs(t[0] * R.m[2][2] - t[2] * R.m[0][2]) > ra + rb)
		return notIntersecting;

	// Test axis L = A2 x B0
	ra = ea[0] * AbsR.m[1][0] + ea[1] * AbsR.m[0][0];
	rb = eb[1] * AbsR.m[2][2] + eb[2] * AbsR.m[2][1];
	if (qAbs(t[1] * R.m[0][0] - t[0] * R.m[1][0]) > ra + rb)
		return notIntersecting;

	// Test axis L = A2 x B1
	ra = ea[0] * AbsR.m[1][1] + ea[1] * AbsR.m[0][1];
	rb = eb[0] * AbsR.m[2][2] + eb[2] * AbsR.m[2][0];
	if (qAbs(t[1] * R.m[0][1] - t[0] * R.m[1][1]) > ra + rb)
		return notIntersecting;

	// Test axis L = A2 x B2
	ra = ea[0] * AbsR.m[1][2] + ea[1] * AbsR.m[0][2];
	rb = eb[0] * AbsR.m[2][1] + eb[1] * AbsR.m[2][0];
	if (qAbs(t[1] * R.m[0][2] - t[0] * R.m[1][2]) > ra + rb)
		return notIntersecting;

	// Since no separating axis is found, the OBBs must be intersecting
	return isIntersecting;
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

void BoxCollider::setHalfExtents( const vec3& halfExtents )
{
	m_boxShape.setHalfExtents(halfExtents);
}

vec3 BoxCollider::getLocalSupportPoint( const vec3& dir, float margin /*= 0*/ ) const
{
	vec3 supportPoint;

	vec3 halfExtents = m_boxShape.getHalfExtents();

	supportPoint.setX(dir.x() < 0 ? -halfExtents.x() - margin : halfExtents.x() + margin);
	supportPoint.setY(dir.y() < 0 ? -halfExtents.y() - margin : halfExtents.y() + margin);
	supportPoint.setZ(dir.z() < 0 ? -halfExtents.z() - margin : halfExtents.z() + margin);

	return supportPoint;
}
