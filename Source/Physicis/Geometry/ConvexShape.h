#pragma once
#include "AbstractShape.h"

class ConvexShape : public AbstractShape
{
public:
	/// Creates a convex shape with the given vertices and faces.
	ConvexShape(const QVector<vec3>& vertices, const QVector<vec3>& faces);

	/// Find the max projection on the axis with brute force
	inline vec3 furthestPoint(vec3& axis) const
	{
		axis.normalize();
		QMap<float, int> indexMap;
		for(int i = 0; i < m_vertices.size(); ++i)
		{
			indexMap[vec3::dotProduct(m_vertices[i], axis)] = i;
		}

		float maxProj = 0.0f;
		foreach(float val, indexMap.keys())
		{
			maxProj = qMax(maxProj, val);
		}

		return m_vertices[indexMap[maxProj]];
	}

	/// Find the max projection on the axis using a hill-climbing algorithm
// 	inline vec3 furthestPoint(vec3& axis) const
// 	{
// 		axis.normalize();
// 		uint next = 0, last = 0, curr = 0;
// 		float p = 0.0f;
// 		float max = vec3::dotProduct(m_vertices[curr], axis);
// 		while (true)
// 		{
// 			if (m_vertexNeighbours[curr].size() == 0)
// 			{
// 				++curr;
// 				continue;
// 			}
// 			for (int i = 0; i < m_vertexNeighbours[curr].size(); ++i)
// 			{
// 				next = m_vertexNeighbours[curr][i];
// 				if (next != last)
// 				{
// 					p = vec3::dotProduct(m_vertices[next], axis);
// 					if (p > max)
// 					{
// 						max = p;
// 						last = curr;
// 						curr = next;
// 						break;
// 					}
// 				}
// 				if (i == m_vertexNeighbours[curr].size() - 1)
// 				{
// 					return m_vertices[curr];
// 				}
// 			}
// 		}
// 	}

private:
	QVector<vec3>          m_vertices;
	//QVector<QVector<uint>> m_vertexNeighbours;
	float m_innerRadius;
	float m_outterRadius;
};

