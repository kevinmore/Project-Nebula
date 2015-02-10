#pragma once

#include "EPAEdge.h"

class EPATriangle
{
friend class EPAPolytope;

public:
	EPATriangle();
	EPATriangle(int indexVertex0, int indexVertex1, int indexVertex2);
	~EPATriangle();

	int getIndexVertex(int i) const 
	{ 
		Q_ASSERT(0 <= i && i < 3);
		return m_indicesVertex[i]; 
	}

	EPAEdge* getEdge(int i)
	{
		Q_ASSERT(0 <= i && i < 3);
		return m_edges[i];
	}

	void setAdjacentEdge(int index, EPAEdge& EPAEdge);
	float getDistSqr() const { return m_distSqrd; }
	bool isObsolete() const { return m_bObsolete; }
	void setObsolete(bool bObsolete) { m_bObsolete = bObsolete; }	
	const vec3& getClosestPoint() const { return m_closestPointToOrigin; }
	bool isClosestPointInternal() const;
	bool isVisibleFromPoint(const vec3& point) const;
	bool computeClosestPointToOrigin(const EPAPolytope& EPAPolytope);
	vec3 getClosestPointToOriginInSupportPntSpace(const QVector<vec3>& supportPoints) const;
	bool doSilhouette(const vec3& w, EPAEdge* edge, EPAPolytope& EPAPolytope);

	bool operator<(const EPATriangle& other) const;

private:
	int m_indicesVertex[3];
	EPATriangle* m_adjacentTriangles[3];
	EPAEdge* m_edges[3];
	bool m_bObsolete;
	float m_det;

	vec3 m_closestPointToOrigin; 

	float m_lambda1; 
	float m_lambda2;

	// squared distance to origin
	float m_distSqrd; // = m_ClosestPointToOrigin.LenghSqr()

	int m_index;
	bool m_bVisible;
};

class EPATriangleComparison 
{
public:
	bool operator() (const EPATriangle* pTriA, const EPATriangle* pTriB) 
	{
		return (pTriA->getDistSqr() > pTriB->getDistSqr());
	}
};