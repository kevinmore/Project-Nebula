#pragma once
#include <Utility/EngineCommon.h>

class EPATriangle;
class EPAEdge
{
	friend class EPATriangle;
	friend class EPAPolytope;

public:
	EPAEdge(EPATriangle* pEPATriangle, int indexLocal, int indexVertex0, int indexVertex1)
	{
		Q_ASSERT(indexLocal >= 0 && indexLocal < 3);
		m_IndexVertex[0] = indexVertex0;
		m_IndexVertex[1] = indexVertex1;
	}

	EPATriangle* m_pEPATriangle; // pointer to owner triangle
	EPAEdge* m_pPairEdge;


	int GetIndexLocal() const { return m_IndexLocal; }
	EPATriangle* GetEPATriangle() const { return m_pEPATriangle; }

	int GetIndexVertex(int i)
	{
		Q_ASSERT( i == 0 || i == 1);
		return m_IndexVertex[i];
	}

private:
	int m_IndexLocal; // 0, 1 or 2 From m_pEPATriangle's point of view, 0, 1, 2 winding order is counter clockwise.

	int m_IndexVertex[2]; // m_IndexVertex[0] is index of starting vertex. m_IndexVertex[1] is index of ending vertex. 
};

