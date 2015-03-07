#include "ConvexHullCollider.h"

ConvexHullCollider::ConvexHullCollider( const vec3& center, const ConvexShape& shape )
	: ICollider(center),
	  m_convexShape(shape),
	  m_scale(Math::Vector3::UNIT_SCALE)
{
	m_colliderType = ICollider::COLLIDER_CONVEXHULL;

	init();
}

ConvexShape ConvexHullCollider::getGeometryShape() const
{
	return m_convexShape;
}

void ConvexHullCollider::init()
{
	ICollider::init();

	// generate the index buffer
	QVector<uint> indices;
	QVector<vec3> faces = m_convexShape.getFaces();
	QVector<vec3> vertices = m_convexShape.getRenderingVertices();

	foreach(vec3 face, faces)
	{
		indices << face.x() << face.y() << face.z();
	}

	MeshPtr mesh(new Mesh("ConvexHull", indices.size(), 0, 0));
	m_meshes << mesh;

	// Create the VAO
	glGenVertexArrays(1, &m_vao);   
	glBindVertexArray(m_vao);

	GLuint indexBufferID, vertexBufferID;
	glGenBuffers(1, &indexBufferID);
	glGenBuffers(1, &vertexBufferID);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices[0]) * indices.size(), indices.data(), GL_STATIC_DRAW);

	GLuint POSITION_LOCATION = glGetAttribLocation(m_renderingEffect->getShaderProgram()->programId(), "Position");
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferID);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices[0]) * vertices.size(), vertices.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(POSITION_LOCATION);
	glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);
	// Make sure the VAO is not changed from the outside
	glBindVertexArray(0);

	m_renderingEffect->setVAO(m_vao);
}

vec3 ConvexHullCollider::getLocalSupportPoint( const vec3& dir, float margin /*= 0*/ ) const
{
	float maxDot = -FLT_MAX;
	vec3 supportPoint;

	for ( int i = 0; i < m_convexShape.getVertices().size(); ++i )
	{
		const vec3 vertex = m_convexShape.getVertices()[i];
		float dot = vec3::dotProduct(vertex, dir);

		if ( dot > maxDot )
		{
			supportPoint = vertex;
			maxDot = dot;
		}
	}

	// apply the scale
	supportPoint.setX(supportPoint.x() * m_scale.x());
	supportPoint.setY(supportPoint.y() * m_scale.y());
	supportPoint.setZ(supportPoint.z() * m_scale.z());

	return supportPoint;
}

BroadPhaseCollisionFeedback ConvexHullCollider::onBroadPhase( ICollider* other )
{
	/*do nothing, this collider is for narrow phase collision detection*/
	return BroadPhaseCollisionFeedback();
}

void ConvexHullCollider::setScale( const vec3& scale )
{
	m_transformMatrix.scale(scale);

	m_scale.setX(m_scale.x() * scale.x());
	m_scale.setY(m_scale.y() * scale.y());
	m_scale.setZ(m_scale.z() * scale.z());
}
