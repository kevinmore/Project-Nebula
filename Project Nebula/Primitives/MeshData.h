#pragma once
#include <Primitives/Vertex.h>
#include <QtOpenGL/QGLBuffer>

class MeshData
{
public:
	Vertex* vertices;
	GLuint numVertices;
	GLushort* indices;
	GLuint numIndices;
	GLuint m_shaderProgramID;
	QGLBuffer vertexBuff;

	MeshData():vertices(0), numVertices(0), indices(0), numIndices(0){};

	void createVertexBuffer()
	{
		vertexBuff = QGLBuffer(QGLBuffer::VertexBuffer);
		vertexBuff.create();
		vertexBuff.bind();
		vertexBuff.setUsagePattern(QGLBuffer::DynamicDraw);
		vertexBuff.allocate(numVertices*sizeof(Vertex));
		vertexBuff.write(0, vertices, numVertices*sizeof(Vertex));
		vertexBuff.release();
	}

	void cleanUp()
	{
		delete [] vertices;
		delete [] indices;
		numVertices = 0;
		numIndices = 0;
	}
};


