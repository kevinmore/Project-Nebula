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
	: m_HalfExtent(vec3(1.0f, 1.0f, 1.0f))
{
	SetColor(1.0, 1.0, 1.0);
}

CollisionObject::CollisionObject( const CollisionObject& other )
{
	m_CollisionObjectType = other.m_CollisionObjectType;
	m_Transform = other.m_Transform;
	m_HalfExtent = other.m_HalfExtent;

	for ( int i = 0; i < 4; i++ )
		m_Color[i] = other.m_Color[i];

	m_Vertices = other.m_Vertices;	
	m_Normals = other.m_Normals;
	m_Faces = other.m_Faces;
	m_Edges = other.m_Edges;

	m_VisualizedPoints = other.m_VisualizedPoints;

}

CollisionObject::~CollisionObject()
{}

bool CollisionObject::Create()
{

	return true;
}

const Transform& CollisionObject::GetTransform() const 
{ 
	return m_Transform; 
}

Transform& CollisionObject::GetTransform() 
{ 
	return m_Transform; 
}

void CollisionObject::SetCollisionObjectType( CollisionObjectType collisionObjectType )
{
	m_CollisionObjectType = collisionObjectType; 
}

vec3 CollisionObject::GetLocalSupportPoint( const vec3& dir, float margin /*= 0*/ ) const
{
	vec3 supportPoint;

	if ( m_CollisionObjectType == Point )
	{
		if ( dir.lengthSquared() > 0.0 )
		{
			vec3 dirN = dir.normalized();
			supportPoint = margin*dirN;
		}
		else
			supportPoint = vec3(margin, 0, 0);
	}
	else if ( m_CollisionObjectType == LineSegment )
	{
		float maxDot = -FLT_MAX;

		for ( int i = 0; i < (int)m_Vertices.size(); i++ )
		{
			const vec3& vertex = m_Vertices[i];
			float dot = vec3::dotProduct(vertex, dir);

			if ( dot > maxDot )
			{
				supportPoint = vertex;
				maxDot = dot;
			}
		}
	}
	else if ( m_CollisionObjectType == Sphere )
	{
		float radius = m_HalfExtent.x();

		if ( dir.lengthSquared() > 0.0 )
			supportPoint = (radius + margin) * dir.normalized();
		else
			supportPoint = vec3(0, radius + margin, 0);
	}
	else if ( m_CollisionObjectType == Box )
	{		
		supportPoint.setX(dir.x() < 0 ? -m_HalfExtent.x() - margin : m_HalfExtent.x() + margin);
		supportPoint.setY(dir.y() < 0 ? -m_HalfExtent.y() - margin : m_HalfExtent.y() + margin);
		supportPoint.setZ(dir.z() < 0 ? -m_HalfExtent.z() - margin : m_HalfExtent.z() + margin);
	}
	else if ( m_CollisionObjectType == Cone )
	{
		float radius = m_HalfExtent.x();
		float halfHeight = 2.0*m_HalfExtent.y();
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
	else if ( m_CollisionObjectType == ConvexHull )
	{
		float maxDot = -FLT_MAX;

		for ( int i = 0; i < (int)m_Vertices.size(); i++ )
		{
			const vec3& vertex = m_Vertices[i];
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
	m_CollisionObjectType = other.m_CollisionObjectType;
	m_Transform = other.m_Transform;
	m_HalfExtent = other.m_HalfExtent;

	for ( int i = 0; i < 4; i++ )
		m_Color[i] = other.m_Color[i];

	m_Vertices = other.m_Vertices;	
	m_Normals = other.m_Normals;
	m_Faces = other.m_Faces;
	m_Edges = other.m_Edges;

	m_VisualizedPoints = other.m_VisualizedPoints;

	return (*this);
}
