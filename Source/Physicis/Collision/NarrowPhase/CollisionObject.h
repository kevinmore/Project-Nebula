#pragma once
#include <Primitives/Component.h>
#include <Primitives/Transform.h>

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
// 	std::vector<vec3>& getNormals() { return m_normals; }
// 	std::vector<TriangleFace>& getFaces() { return m_faces; }
// 	std::vector<Edge>& getEdges() { return m_edges; }

	float getMargin() const { return m_margin; }
	void setMargin(float margin) { m_margin = margin; }

	Transform getTransform() const;

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
// 	std::vector<vec3> m_normals;
// 	std::vector<TriangleFace> m_faces;
// 	std::vector<Edge> m_edges;

	// For visualization
	std::vector<vec3> m_visualizedPoints;
};

