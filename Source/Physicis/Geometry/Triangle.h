#pragma once

#include "Edge.h"

class Triangle
{
friend class Polytope;

public:
	Triangle();
	Triangle(int indexVertex0, int indexVertex1, int indexVertex2);
	~Triangle();

	int getIndexVertex(int i) const 
	{ 
		assert(0 <= i && i < 3);
		return m_indicesVertex[i]; 
	}

	Edge* getEdge(int i)
	{
		assert(0 <= i && i < 3);
		return m_edges[i];
	}

	void setAdjacentEdge(int index, Edge& Edge);
	float getDistSqrd() const { return m_distSqrd; }
	bool isObsolete() const { return m_bObsolete; }
	void setObsolete(bool bObsolete) { m_bObsolete = bObsolete; }	
	const vec3& getClosestPoint() const { return m_closestPointToOrigin; }
	bool isClosestPointInternal() const;
	bool isVisibleFromPoint(const vec3& point) const;
	bool computeClosestPointToOrigin(const Polytope& Polytope);
	vec3 getClosestPointToOriginInSupportPntSpace(const QVector<vec3>& supportPoints) const;
	bool doSilhouette(const vec3& w, Edge* edge, Polytope& Polytope);

	bool operator<(const Triangle& other) const;

private:
	int m_indicesVertex[3];
	Triangle* m_adjacentTriangles[3];
	Edge* m_edges[3];
	bool m_bObsolete;
	float m_det;

	vec3 m_closestPointToOrigin; 

	float m_lambda1; 
	float m_lambda2;

	// squared distance to origin
	float m_distSqrd; // = m_closestPointToOrigin.LenghSqr()
};

class TriangleComparison 
{
public:
	bool operator() (const Triangle* pTriA, const Triangle* pTriB) 
	{
		return (pTriA->getDistSqrd() > pTriB->getDistSqrd());
	}
};