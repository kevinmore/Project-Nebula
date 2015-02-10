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
	void SetFaceIndex(int indexFace)
	{
		std::vector<int>::iterator iter = std::find(m_IndexFaces.begin(), m_IndexFaces.end(), indexFace);

		if ( iter == m_IndexFaces.end() )
		{
			m_IndexFaces.push_back(indexFace);
		}
	}

	const std::vector<int>& GetFaceIndeces() const
	{		
		return m_IndexFaces;
	}

	void SetEdgeIndex(int indexEdge)
	{
		std::vector<int>::iterator iter = std::find(m_IndexEdges.begin(), m_IndexEdges.end(), indexEdge);

		if ( iter == m_IndexEdges.end() )
		{
			m_IndexEdges.push_back(indexEdge);
		}
	}

	const std::vector<int>& GetEdgeIndeces() const
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
	int GetVertexIndex(int i) const 
	{
		Q_ASSERT( 0 <= i && i <= 1 );

		return m_IndexVrx[i];
	}

	int GetIndex() const { return m_Index; }
	void SetIndex(int index) { m_Index = index; }

	int GetTriangleIndex(int i) const 
	{ 
		Q_ASSERT( 0 <= i && i <= 1 );
		return m_IndexTriangle[i]; 
	}

	void SetTriangleIndex(int i, int indexTriangle) 
	{
		Q_ASSERT( 0 <= i && i <= 1 );
		m_IndexTriangle[i] = indexTriangle;
	}

	int GetTheOtherVertexIndex(int indexVert)
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
	int GetVertexIndex(int i) const 
	{
		Q_ASSERT( 0 <= i && i < 3 );
		return m_IndexVrx[i];
	}

	void SetVertexIndex(int i, int vertexIndex)
	{
		Q_ASSERT( 0 <= i && i < 3 );
		m_IndexVrx[i] = vertexIndex;
	}

	int GetEdgeIndex(int i) const 
	{
		Q_ASSERT( 0 <= i && i < 3 );
		return m_IndexEdge[i];
	}

	void SetEdgeIndex(int i, int edgeIndex) 
	{
		Q_ASSERT( 0 <= i && i < 3 );
		m_IndexEdge[i] = edgeIndex;
	}

	bool GetWindingOrderEdge(int i) const
	{
		Q_ASSERT( 0 <= i && i < 3 );
		return m_WindingOrderEdge[i];
	}

	bool GetWindingOrderEdgeByGlobalEdgeIndex(int indexEdge) const
	{
		for ( int i = 0; i < 3; i++ )
		{
			if ( m_IndexEdge[i] == indexEdge )
				return m_WindingOrderEdge[i];
		}

		Q_ASSERT(false); // should not reach here. 
		return true;
	}

	void SetWindingOrderEdge(int i, bool bWindingOderEdge)
	{
		Q_ASSERT( 0 <= i && i < 3 );
		m_WindingOrderEdge[i] = bWindingOderEdge;
	}

	int GetIndex() const { return m_Index; }
	void SetIndex(int index) { m_Index = index; }

	float* PlaneEquation() { return m_PlaneEqn; }
	const float* PlaneEquation() const { return m_PlaneEqn; }

	vec3 GetNormal() const { return vec3(m_PlaneEqn[0], m_PlaneEqn[1], m_PlaneEqn[2]); }

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

	bool Create();

	CollisionObjectType GetCollisionObjectType() const { return m_CollisionObjectType; }
	void SetCollisionObjectType(CollisionObjectType collisionObjectType);

	void SetSize(float x, float y, float z) { m_HalfExtent = vec3(x/2.0f, y/2.0f, z/2.0f); }
	void SetColor(float r, float g, float b) { m_Color[0] = r; m_Color[1] = g; m_Color[2] = b; m_Color[3] = 1.0; }

	std::vector<Vertex>& GetVertices() { return m_Vertices; }
	const std::vector<Vertex>& GetVertices() const { return m_Vertices; }

	std::vector<vec3>& GetNormals() { return m_Normals; }
	const std::vector<vec3>& GetNormals() const { return m_Normals; }

	std::vector<TriangleFace>& GetFaces() { return m_Faces; }
	const std::vector<TriangleFace>& GetFaces() const { return m_Faces; }

	std::vector<Edge>& GetEdges() { return m_Edges; }
	const std::vector<Edge>& GetEdges() const { return m_Edges; }

	const Transform& GetTransform() const;
	Transform& GetTransform();

	vec3 GetSize() const { return 2.0 * m_HalfExtent; }

	vec3 GetLocalSupportPoint(const vec3& dir, float margin = 0) const;

	CollisionObject& operator=(const CollisionObject& other);

protected:
	CollisionObjectType m_CollisionObjectType; 

	// transforms local to world. 
	Transform m_Transform;

	vec3 m_HalfExtent;

	float m_Color[4];

	// For ConvexHull
	std::vector<Vertex> m_Vertices;	
	std::vector<vec3> m_Normals;
	std::vector<TriangleFace> m_Faces;
	std::vector<Edge> m_Edges;

	// For visualization
	std::vector<vec3> m_VisualizedPoints;
};

