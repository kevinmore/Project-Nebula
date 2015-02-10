#pragma once
#include <Primitives/Component.h>
#include <Physicis/Transform.h>

class Vertex : public vec3
{
public:
	Vertex() {}
	Vertex(float x, float y, float z) : vec3(x, y, z) {}
	Vertex(const Vertex& other) : vec3(other) 
	{
		m_IndexFaces = other.m_IndexFaces;
	}

private:
	std::vector<int> m_IndexFaces;
	std::vector<int> m_IndexEdges;

public:
	void setFaceIndex(int indexFace)
	{
		std::vector<int>::iterator iter = std::find(m_IndexFaces.begin(), m_IndexFaces.end(), indexFace);

		if ( iter == m_IndexFaces.end() )
		{
			m_IndexFaces.push_back(indexFace);
		}
	}

	const std::vector<int>& getFaceIndeces() const
	{		
		return m_IndexFaces;
	}

	void setEdgeIndex(int indexEdge)
	{
		std::vector<int>::iterator iter = std::find(m_IndexEdges.begin(), m_IndexEdges.end(), indexEdge);

		if ( iter == m_IndexEdges.end() )
		{
			m_IndexEdges.push_back(indexEdge);
		}
	}

	const std::vector<int>& getEdgeIndeces() const
	{		
		return m_IndexEdges;
	}

	Vertex& operator=(const Vertex& other)
	{
		vec3::operator=(other);
		m_IndexFaces = other.m_IndexFaces;

		return (*this);
	}
};

//----------
class Edge
//----------
{
public:
	Edge() : m_bFlag(false)
	{ 
		m_IndexVrx[0] = -1; 
		m_IndexVrx[1] = -1; 
		m_IndexTriangle[0] = -1;
		m_IndexTriangle[1] = -1;
		m_Index = -1;
	}

	Edge(int indexVrx0, int indexVrx1) : m_bFlag(false)
	{ 
		m_IndexVrx[0] = indexVrx0; 
		m_IndexVrx[1] = indexVrx1;
		m_IndexTriangle[0] = -1;
		m_IndexTriangle[1] = -1;
		m_Index = -1; 
	}

	Edge(const Edge& other) 
	{
		for ( int i = 0; i < 2; i++ )
		{
			m_IndexVrx[i] = other.m_IndexVrx[i];
			m_IndexTriangle[i] = other.m_IndexTriangle[i];
		}

		m_bFlag = other.m_bFlag;

		m_Index = other.m_Index;
	}

	virtual ~Edge() { }

	bool m_bFlag; 

protected:
	int m_Index;
	int m_IndexVrx[2];
	int m_IndexTriangle[2];

public:
	int getVertexIndex(int i) const 
	{
		Q_ASSERT( 0 <= i && i <= 1 );

		return m_IndexVrx[i];
	}

	int getIndex() const { return m_Index; }
	void setIndex(int index) { m_Index = index; }

	int getTriangleIndex(int i) const 
	{ 
		Q_ASSERT( 0 <= i && i <= 1 );
		return m_IndexTriangle[i]; 
	}

	void setTriangleIndex(int i, int indexTriangle) 
	{
		Q_ASSERT( 0 <= i && i <= 1 );
		m_IndexTriangle[i] = indexTriangle;
	}

	int getTheOtherVertexIndex(int indexVert)
	{
		Q_ASSERT(indexVert == m_IndexVrx[0] || indexVert == m_IndexVrx[1]);

		if ( indexVert == m_IndexVrx[0] )
			return m_IndexVrx[1];
		else
			return m_IndexVrx[0];
	}

	bool operator==(const Edge& other)
	{
		if ( ( m_IndexVrx[0] == other.m_IndexVrx[0] && m_IndexVrx[1] == other.m_IndexVrx[1] ) ||
			( m_IndexVrx[0] == other.m_IndexVrx[1] && m_IndexVrx[1] == other.m_IndexVrx[0] ) )
			return true;
		else
			return false;
	}

	Edge& operator=(const Edge& other)
	{
		for ( int i = 0; i < 2; i++ )
		{
			m_IndexVrx[i] = other.m_IndexVrx[i];
			m_IndexTriangle[i] = other.m_IndexTriangle[i];
		}

		m_bFlag = other.m_bFlag;
		m_Index = other.m_Index;

		return (*this);
	}
};

//------------------
class TriangleFace
//------------------
{
public:
	TriangleFace();
	TriangleFace(const TriangleFace& other);
	virtual ~TriangleFace();

	bool m_bFlag; 

protected:
	int m_Index;
	int m_IndexVrx[3];
	int m_IndexEdge[3];

	// If true, a vector formed by two points starting from CEdge::m_IndexVrx[0] and ending at CEdge::m_IndexVrx[1] will be right direction
	// in terms of normal vector of this triangle face. The right direction means three vectors will make the counter-clock-wise orientation
	// around normal vector. If false, swap the two points and it will create the right direction. 
	bool m_WindingOrderEdge[3]; 

	float m_PlaneEqn[4];

public:
	int getVertexIndex(int i) const 
	{
		Q_ASSERT( 0 <= i && i < 3 );
		return m_IndexVrx[i];
	}

	void setVertexIndex(int i, int vertexIndex)
	{
		Q_ASSERT( 0 <= i && i < 3 );
		m_IndexVrx[i] = vertexIndex;
	}

	int getEdgeIndex(int i) const 
	{
		Q_ASSERT( 0 <= i && i < 3 );
		return m_IndexEdge[i];
	}

	void setEdgeIndex(int i, int edgeIndex) 
	{
		Q_ASSERT( 0 <= i && i < 3 );
		m_IndexEdge[i] = edgeIndex;
	}

	bool getWindingOrderEdge(int i) const
	{
		Q_ASSERT( 0 <= i && i < 3 );
		return m_WindingOrderEdge[i];
	}

	bool getWindingOrderEdgeByGlobalEdgeIndex(int indexEdge) const
	{
		for ( int i = 0; i < 3; i++ )
		{
			if ( m_IndexEdge[i] == indexEdge )
				return m_WindingOrderEdge[i];
		}

		Q_ASSERT(false); // should not reach here. 
		return true;
	}

	void setWindingOrderEdge(int i, bool bWindingOderEdge)
	{
		Q_ASSERT( 0 <= i && i < 3 );
		m_WindingOrderEdge[i] = bWindingOderEdge;
	}

	int getIndex() const { return m_Index; }
	void setIndex(int index) { m_Index = index; }

	float* planeEquation() { return m_PlaneEqn; }
	const float* planeEquation() const { return m_PlaneEqn; }

	vec3 getNormal() const { return vec3(m_PlaneEqn[0], m_PlaneEqn[1], m_PlaneEqn[2]); }

	TriangleFace& operator=(const TriangleFace& other);
};

class CollisionObject : public Component
{
public:

	enum CollisionObjectType 
	{ 
		None, 
		Point, 
		LineSegment,
		Box, 
		Sphere, 
		Cone, 
		Capsule,
		Cylinder,
		ConvexHull 
	};

	CollisionObject();
	CollisionObject(const CollisionObject& other);
	~CollisionObject();

	virtual QString className() { return "CollisionObject"; }
	virtual void render(const float currentTime) {/*do noting*/}

	CollisionObjectType getCollisionObjectType() const { return m_collisionObjectType; }
	void setCollisionObjectType(CollisionObjectType collisionObjectType);

	void setSize(float x, float y, float z) { m_halfExtent = vec3(x/2.0f, y/2.0f, z/2.0f); }
	void setColor(float r, float g, float b) { m_color[0] = r; m_color[1] = g; m_color[2] = b; m_color[3] = 1.0; }

	std::vector<Vertex>& getVertices() { return m_vertices; }
	const std::vector<Vertex>& getVertices() const { return m_vertices; }

	std::vector<vec3>& getNormals() { return m_normals; }
	const std::vector<vec3>& getNormals() const { return m_normals; }

	std::vector<TriangleFace>& getFaces() { return m_faces; }
	const std::vector<TriangleFace>& getFaces() const { return m_faces; }

	std::vector<Edge>& getEdges() { return m_edges; }
	const std::vector<Edge>& getEdges() const { return m_edges; }

	float getMargin() const { return m_margin; }
	void setMargin(float margin) { m_margin = margin; }

	const Transform& getTransform() const;
	Transform& getTransform();

	vec3 getSize() const { return 2.0 * m_halfExtent; }

	vec3 getLocalSupportPoint(const vec3& dir, float margin = 0) const;

	CollisionObject& operator=(const CollisionObject& other);

protected:
	CollisionObjectType m_collisionObjectType; 

	// transforms local to world. 
	Transform m_transform;

	vec3 m_halfExtent;

	float m_color[4];

	float m_margin;

	// For ConvexHull
	std::vector<Vertex> m_vertices;	
	std::vector<vec3> m_normals;
	std::vector<TriangleFace> m_faces;
	std::vector<Edge> m_edges;

	// For visualization
	std::vector<vec3> m_visualizedPoints;
};

