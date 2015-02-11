#include "ConvexShape.h"

ConvexShape::ConvexShape( const QVector<vec3>& vertices, const QVector<vec3>& faces )
	: IShape(IShape::CONVEXBODY, vec3(0, 0, 0)),
	  m_vertices(vertices)
{
	//QVector<vec2> edges;
	//QVector<QVector<uint>> vertexNeighbours;
	/* Calculate Properties */
	//Find Edges

	//Iterate over each face and for each next face in the list, check if they
	//share two vertices, this defines an edge.
// 	for(int i = 0; i < faces.size() - 1; ++i)
// 		for(int j = i + 1; j < faces.size(); ++j)
// 	{
// 		uint fCount = 0;
// 		vec2 edge;
// 		vec3 face1 = faces[i];
// 		vec3 face2 = faces[j];
// 
// 		for (int x = 0; x < 3; ++x)
// 			for(int y = 0; y < 3; ++y)
// 		{
// 			if (face1[x] == face2[y])
// 			{
// 				edge[fCount] = face1[x];
// 				++fCount;
// 
// 				if (fCount > 1)
// 				{
// 					edges << edge;
// 					fCount = 0;
// 					continue;
// 				}
// 			}
// 		}
// 		
// 	}


	//Find Vertex Neighbours
	//For all vertices, check if two edges share this vertex. If they do and it
	//isn't vertex 0, append the other vertices of these edge to the neighbor list
// 	for (int vi = 0; vi < vertices.size(); ++vi)
// 	{
// 		QVector<uint> neighbours;
// 		for (int ei = 0; ei < edges.size(); ++ei)
// 			for(uint i = 0; i < 2; ++i)
// 		{
// 			if (edges[ei][i] == vi && edges[ei][(i + 1) % 2] != 0)
// 			{
// 				neighbours << edges[ei][(i + 1) % 2];
// 			}
// 		}
// 		//if(!neighbours.isEmpty())
// 			m_vertexNeighbours << neighbours;
// 	}

	//Find the innerRadius
	//For each face, calculate its distance from the particle's center and find the min
	float minDistance = 100.0f;
	for (int i = 0; i < faces.size(); ++i)
	{
		vec3 p = vertices[faces[i][0]];

		vec3 a(vertices[faces[i][1]][0] - p[0],
			   vertices[faces[i][1]][1] - p[1],
			   vertices[faces[i][1]][2] - p[2]);

		vec3 b(vertices[faces[i][2]][0] - p[0],
			   vertices[faces[i][2]][1] - p[1],
			   vertices[faces[i][2]][2] - p[2]);

		vec3 normal = vec3::crossProduct(a, b).normalized();

		float faceDistance = fabs(vec3::dotProduct(normal, p));

		minDistance = qMin(minDistance, faceDistance);
	}

	m_innerRadius = minDistance;

	//Find the circumRadius
	//It's just the farthest vertex from the particle's center
	float maxDistance = 0.0f;
	for (int i = 0; i < vertices.size(); ++i)
	{
		maxDistance = qMax(maxDistance, vertices[i].lengthSquared());
	}

	m_outterRadius = qSqrt(maxDistance);
}

