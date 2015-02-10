#include "CollisionObject.h"

TriangleFace::TriangleFace() : m_bFlag(false)
{
	for ( int i = 0; i < 3; i++ )
	{
		m_IndexVrx[i] = -1;
		m_IndexEdge[i] = -1;			
	}

	m_Index = -1;
}

TriangleFace::TriangleFace(const TriangleFace& other)
{
	for ( int i = 0; i < 3; i++ )
	{
		m_IndexVrx[i] = other.m_IndexVrx[i];
		m_IndexEdge[i] = other.m_IndexEdge[i];
	}

	for ( int i = 0; i < 4; i++ )
		m_PlaneEqn[i] = other.m_PlaneEqn[i];

	m_bFlag = other.m_bFlag;
	m_Index = other.m_Index;
}

TriangleFace::~TriangleFace() 
{
}

TriangleFace& TriangleFace::operator=(const TriangleFace& other)
{
	for ( int i = 0; i < 3; i++ )
	{
		m_IndexVrx[i] = other.m_IndexVrx[i];
		m_IndexEdge[i] = other.m_IndexEdge[i];
	}

	for ( int i = 0; i < 4; i++ )
		m_PlaneEqn[i] = other.m_PlaneEqn[i];

	m_Index = other.m_Index;
	m_bFlag = other.m_bFlag;

	return (*this);
}


CollisionObject::CollisionObject()
	: m_halfExtent(vec3(1.0f, 1.0f, 1.0f)),
	  m_margin(0.0001f)
{
	setColor(1.0, 1.0, 1.0);
}

CollisionObject::CollisionObject( const CollisionObject& other )
{
	m_collisionObjectType = other.m_collisionObjectType;
	m_transform = other.m_transform;
	m_halfExtent = other.m_halfExtent;
	m_margin = other.m_margin;

	for ( int i = 0; i < 4; i++ )
		m_color[i] = other.m_color[i];

	m_vertices = other.m_vertices;	
	m_normals = other.m_normals;
	m_faces = other.m_faces;
	m_edges = other.m_edges;

	m_visualizedPoints = other.m_visualizedPoints;

}

CollisionObject::~CollisionObject()
{}

const Transform& CollisionObject::getTransform() const 
{ 
	return m_transform; 
}

Transform& CollisionObject::getTransform() 
{ 
	return m_transform; 
}

void CollisionObject::setCollisionObjectType( CollisionObjectType collisionObjectType )
{
	m_collisionObjectType = collisionObjectType; 
}

vec3 CollisionObject::getLocalSupportPoint( const vec3& dir, float margin /*= 0*/ ) const
{
	vec3 supportPoint;

	if ( m_collisionObjectType == Point )
	{
		if ( dir.lengthSquared() > 0.0 )
		{
			vec3 dirN = dir.normalized();
			supportPoint = margin*dirN;
		}
		else
			supportPoint = vec3(margin, 0, 0);
	}
	else if ( m_collisionObjectType == LineSegment )
	{
		float maxDot = -FLT_MAX;

		for ( int i = 0; i < (int)m_vertices.size(); i++ )
		{
			const vec3& vertex = m_vertices[i];
			float dot = vec3::dotProduct(vertex, dir);

			if ( dot > maxDot )
			{
				supportPoint = vertex;
				maxDot = dot;
			}
		}
	}
	else if ( m_collisionObjectType == Sphere )
	{
		float radius = m_halfExtent.x();

		if ( dir.lengthSquared() > 0.0 )
			supportPoint = (radius + margin) * dir.normalized();
		else
			supportPoint = vec3(0, radius + margin, 0);
	}
	else if ( m_collisionObjectType == Box )
	{		
		supportPoint.setX(dir.x() < 0 ? -m_halfExtent.x() - margin : m_halfExtent.x() + margin);
		supportPoint.setY(dir.y() < 0 ? -m_halfExtent.y() - margin : m_halfExtent.y() + margin);
		supportPoint.setZ(dir.z() < 0 ? -m_halfExtent.z() - margin : m_halfExtent.z() + margin);
	}
	else if ( m_collisionObjectType == Cone )
	{
		float radius = m_halfExtent.x();
		float halfHeight = 2.0*m_halfExtent.y();
		float sinTheta = radius / (sqrt(radius * radius + 4 * halfHeight * halfHeight));
		const vec3& v = dir;
		float sinThetaTimesLengthV = sinTheta * v.length();

		if ( v.y() >= sinThetaTimesLengthV) {
			supportPoint = vec3(0.0, halfHeight, 0.0);
		}
		else {
			float projectedLength = sqrt(v.x() * v.x() + v.z() * v.z());
			if (projectedLength > 1e-10) {
				float d = radius / projectedLength;
				supportPoint = vec3(v.x() * d, -halfHeight, v.z() * d);
			}
			else {
				supportPoint = vec3(radius, -halfHeight, 0.0);
			}
		}

		// Add the margin to the support point
		if (margin != 0.0) {
			vec3 unitVec(0.0, -1.0, 0.0);
			if (v.lengthSquared() > 1e-10 * 1e-10) {
				unitVec = v.normalized();
			}
			supportPoint += unitVec * margin;
		}
	}
	else if ( m_collisionObjectType == ConvexHull )
	{
		float maxDot = -FLT_MAX;

		for ( int i = 0; i < (int)m_vertices.size(); i++ )
		{
			const vec3& vertex = m_vertices[i];
			float dot = vec3::dotProduct(vertex, dir);

			if ( dot > maxDot )
			{
				supportPoint = vertex;
				maxDot = dot;
			}
		}
	}



	return supportPoint;
}

CollisionObject& CollisionObject::operator=( const CollisionObject& other )
{
	m_collisionObjectType = other.m_collisionObjectType;
	m_transform = other.m_transform;
	m_halfExtent = other.m_halfExtent;
	m_margin = other.m_margin;

	for ( int i = 0; i < 4; i++ )
		m_color[i] = other.m_color[i];

	m_vertices = other.m_vertices;	
	m_normals = other.m_normals;
	m_faces = other.m_faces;
	m_edges = other.m_edges;

	m_visualizedPoints = other.m_visualizedPoints;

	return (*this);
}
