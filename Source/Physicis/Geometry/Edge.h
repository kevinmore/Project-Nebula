#pragma once
#include <Utility/EngineCommon.h>

class Triangle;
class Edge
{
	friend class Triangle;
	friend class Polytope;

public:
	Edge(Triangle* pTriangle, int indexLocal, int indexVertex0, int indexVertex1)
		: m_pTriangle(pTriangle),
		  m_indexLocal(indexLocal)
	{
		assert(indexLocal >= 0 && indexLocal < 3);
		m_indexVertex[0] = indexVertex0;
		m_indexVertex[1] = indexVertex1;
	}

	Triangle* m_pTriangle; // pointer to owner triangle
	Edge* m_pPairEdge;


	int getIndexLocal() const { return m_indexLocal; }
	Triangle* getTriangle() const { return m_pTriangle; }

	int getIndexVertex(int i)
	{
		assert( i == 0 || i == 1);
		return m_indexVertex[i];
	}

private:
	int m_indexLocal; // 0, 1 or 2 From m_pTriangle's point of view, 0, 1, 2 winding order is counter clockwise.

	int m_indexVertex[2]; // m_IndexVertex[0] is index of starting vertex. m_IndexVertex[1] is index of ending vertex. 
};

