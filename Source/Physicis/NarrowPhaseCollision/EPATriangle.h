#pragma once

#include "EPAEdge.h"

class EPATriangle
{
friend class EPAPolytope;

public:
	EPATriangle();
	EPATriangle(int indexVertex0, int indexVertex1, int indexVertex2);
	~EPATriangle();

	int GetIndexVertex(int i) const 
	{ 
		Q_ASSERT(0 <= i && i < 3);
		return m_IndicesVertex[i]; 
	}

	EPAEdge* GetEdge(int i)
	{
		Q_ASSERT(0 <= i && i < 3);
		return m_Edges[i];
	}

	void SetAdjacentEdge(int index, EPAEdge& EPAEdge);
	float GetDistSqr() const { return m_DistSqr; }
	bool IsObsolete() const { return m_bObsolete; }
	void SetObsolete(bool bObsolete) { m_bObsolete = bObsolete; }	
	const vec3& GetClosestPoint() const { return m_ClosestPointToOrigin; }
	bool IsClosestPointInternal() const;
	bool IsVisibleFromPoint(const vec3& point) const;
	bool ComputeClosestPointToOrigin(const EPAPolytope& EPAPolytope);
	vec3 GetClosestPointToOriginInSupportPntSpace(const QVector<vec3>& supportPoints) const;
	bool DoSilhouette(const vec3& w, EPAEdge* edge, EPAPolytope& EPAPolytope);

	bool operator<(const EPATriangle& other) const;

private:
	int m_IndicesVertex[3];
	EPATriangle* m_AdjacentTriangles[3];
	EPAEdge* m_Edges[3];
	bool m_bObsolete;
	float m_Det;

	vec3 m_ClosestPointToOrigin; 

	float m_Lambda1; 
	float m_Lambda2;

	// squared distance to origin
	float m_DistSqr; // = m_ClosestPointToOrigin.LenghSqr()

	int m_Index;
	bool m_bVisible;
};

class EPATriangleComparison 
{
public:
	bool operator() (const EPATriangle* pTriA, const EPATriangle* pTriB) 
	{
		return (pTriA->GetDistSqr() > pTriB->GetDistSqr());
	}
};