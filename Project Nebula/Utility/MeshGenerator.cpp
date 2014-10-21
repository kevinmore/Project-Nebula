#include <Utility/MeshGenerator.h>

#define  NUM_ARRAY_ELEMENTS(a) sizeof(a)/sizeof(*a)

void MeshGenerator::generateNormals( MeshData &mesh )
{
	QVector<vec3>* normal_buffer = new QVector<vec3>[mesh.numVertices];

	for (uint i = 0; i < mesh.numIndices; i+=3)
	{
		// get the three vertices that make the faces
		vec3 p1 = mesh.vertices[mesh.indices[i+0]].postition;
		vec3 p2 = mesh.vertices[mesh.indices[i+1]].postition;
		vec3 p3 = mesh.vertices[mesh.indices[i+2]].postition;
		
		vec3 v1 = p2 - p1;
		vec3 v2 = p3 - p1;
		vec3 normal = QVector3D::crossProduct(v1, v2);
		normal.normalize();
		mesh.vertices[mesh.indices[i]].normal = normal;

		
		// Store the face's normal for each of the vertices that make up the face.
		normal_buffer[mesh.indices[i+0]].push_back( normal );
		normal_buffer[mesh.indices[i+1]].push_back( normal );
		normal_buffer[mesh.indices[i+2]].push_back( normal );
	}

	// Now loop through each vertex, and average out all the normals stored.
	for( uint i = 0; i < mesh.numVertices; ++i )
	{
		for( int j = 0; j < normal_buffer[i].size(); ++j )
			mesh.vertices[i].normal += normal_buffer[i][j];

		mesh.vertices[i].normal /= normal_buffer[i].size();
	}

}

/*
ShapeData ShapeGenerator::makeTriangle()
{
	ShapeData ret;

	Vertex verts[] = {
		// first triangle
		glm::vec3(-1.0f, -1.0f, +0.5f), // pos
		glm::vec4(1.0f, 0.0f, 0.0f, 1.0f), // color

		glm::vec3(+0.0f, +1.0f, -1.0f),
		glm::vec4(0.0f, 1.0f, 0.0f, 1.0f),

		glm::vec3(+1.0f, -1.0f, +0.5f),
		glm::vec4(0.0f, 0.0f, 1.0f, 1.0f),
	};

	ret.numVertices = NUM_ARRAY_ELEMENTS(verts);
	ret.vertices = new Vertex[ret.numVertices];
	memcpy(ret.vertices, verts, sizeof(verts));

	// indices
	GLushort indices[] = {
		0, 1, 2
	};

	ret.numIndices = NUM_ARRAY_ELEMENTS(indices);
	ret.indices = new GLushort[ret.numIndices];
	memcpy(ret.indices, indices, sizeof(indices));

	return ret;
}

ShapeData ShapeGenerator::makeCube()
{
	ShapeData ret;
	Vertex stackVerts[] = {
		vec3(-1.0f, +1.0f, +1.0f), // 0
		vec4(+1.0f, +0.0f, +0.0f, 1.0f), // Colour
		vec3(+1.0f, +1.0f, +1.0f), // 1
		vec4(+0.0f, +1.0f, +0.0f, 1.0f), // Colour
		vec3(+1.0f, +1.0f, -1.0f), // 2
		vec4(+0.0f, +0.0f, +1.0f, 1.0f), // Colour
		vec3(-1.0f, +1.0f, -1.0f), // 3
		vec4(+1.0f, +1.0f, +1.0f, 1.0f), // Colour

		vec3(-1.0f, +1.0f, -1.0f), // 4
		vec4(+1.0f, +0.0f, +1.0f, 1.0f), // Colour
		vec3(+1.0f, +1.0f, -1.0f), // 5
		vec4(+0.0f, +0.5f, +0.2f, 1.0f), // Colour
		vec3(+1.0f, -1.0f, -1.0f), // 6
		vec4(+0.8f, +0.6f, +0.4f, 1.0f), // Colour
		vec3(-1.0f, -1.0f, -1.0f), // 7
		vec4(+0.3f, +1.0f, +0.5f, 1.0f), // Colour

		vec3(+1.0f, +1.0f, -1.0f), // 8
		vec4(+0.2f, +0.5f, +0.2f, 1.0f), // Colour
		vec3(+1.0f, +1.0f, +1.0f), // 9
		vec4(+0.9f, +0.3f, +0.7f, 1.0f), // Colour
		vec3(+1.0f, -1.0f, +1.0f), // 10
		vec4(+0.3f, +0.7f, +0.5f, 1.0f), // Colour
		vec3(+1.0f, -1.0f, -1.0f), // 11
		vec4(+0.5f, +0.7f, +0.5f, 1.0f), // Colour

		vec3(-1.0f, +1.0f, +1.0f), // 12
		vec4(+0.7f, +0.8f, +0.2f, 1.0f), // Colour
		vec3(-1.0f, +1.0f, -1.0f), // 13
		vec4(+0.5f, +0.7f, +0.3f, 1.0f), // Colour
		vec3(-1.0f, -1.0f, -1.0f), // 14
		vec4(+0.4f, +0.7f, +0.7f, 1.0f), // Colour
		vec3(-1.0f, -1.0f, -1.0f), // 15
		vec4(+0.2f, +0.5f, +1.0f, 1.0f), // Colour

		vec3(+1.0f, +1.0f, +1.0f), // 16
		vec4(+0.6f, +1.0f, +0.7f, 1.0f), // Colour
		vec3(-1.0f, +1.0f, +1.0f), // 17
		vec4(+0.6f, +0.4f, +0.8f, 1.0f), // Colour
		vec3(-1.0f, -1.0f, +1.0f), // 18
		vec4(+0.2f, +0.8f, +0.7f, 1.0f), // Colour
		vec3(+1.0f, -1.0f, +1.0f), // 19
		vec4(+0.2f, +0.7f, +1.0f, 1.0f), // Colour

		vec3(+1.0f, -1.0f, -1.0f), // 20
		vec4(+0.8f, +0.3f, +0.7f, 1.0f), // Colour
		vec3(-1.0f, -1.0f, -1.0f), // 21
		vec4(+0.8f, +0.9f, +0.5f, 1.0f), // Colour
		vec3(-1.0f, -1.0f, +1.0f), // 22
		vec4(+0.5f, +0.8f, +0.5f, 1.0f), // Colour
		vec3(+1.0f, -1.0f, +1.0f), // 23
		vec4(+0.9f, +1.0f, +0.2f, 1.0f), // Colour
	};
	ret.numVertices = NUM_ARRAY_ELEMENTS(stackVerts);
	ret.vertices = new Vertex[ret.numVertices];
	memcpy(ret.vertices, stackVerts, sizeof(stackVerts));

	unsigned short stackIndices[] = {
		0,  1,  2,  0,  2,  3, // Top
		4,  5,  6,  4,  6,  7, // Front
		8,  9, 10,  8, 10, 11, // Right
		12, 13, 14, 12, 14, 15, // Left
		16, 17, 18, 16, 18, 19, // Back
		20, 22, 21, 20, 23, 22, // Bottom
	};

	ret.numIndices = NUM_ARRAY_ELEMENTS(stackIndices);
	ret.indices = new GLushort[ret.numIndices];
	memcpy(ret.indices, stackIndices, sizeof(stackIndices));

	

	return ret;
}

*/

MeshData MeshGenerator::makePalm(vec4 &color)
{
	MeshData ret;
	Vertex stackVerts[] = {
		vec3(-0.03f, +0.05f, 0.01f), // 0
		color,
		vec3(1.0f, 0.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(-0.03f, +0.05f, -0.01f), // 1
		color,
		vec3(1.0f, 0.0f, 1.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(+0.03f, +0.05f, -0.01f), // 2
		color,
		vec3(-1.0f, -1.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(+0.03f, +0.05f, 0.01f), // 3
		color,
		vec3(0.0f, 0.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(-0.03f, 0.0f, 0.01f), // 4
		color,
		vec3(1.0f, 1.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(-0.03f, 0.0f, -0.01f), // 5
		color,
		vec3(1.0f, 1.0f, 1.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(+0.03f, 0.0f, -0.01f), // 6
		color,
		vec3(0.0f, 1.0f, 1.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(+0.03f, 0.0f, 0.01f), // 7
		color,
		vec3(0.0f, 1.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData()
	};

	ret.numVertices = NUM_ARRAY_ELEMENTS(stackVerts);
	ret.vertices = new Vertex[ret.numVertices];
	memcpy(ret.vertices, stackVerts, sizeof(stackVerts));

	unsigned short stackIndices[] = {
		0, 3, 2, 2, 1, 0, // Top
		0, 4, 7, 7, 3, 0, // Front
		3, 7, 6, 6, 2, 3, // Right
		1, 5, 4, 4, 0, 1, // Left
		2, 6, 5, 5, 1, 2, // Back
		4, 5, 6, 6, 7, 4 // Bottom
	};		
	ret.numIndices = NUM_ARRAY_ELEMENTS(stackIndices);
	ret.indices = new GLushort[ret.numIndices];
	memcpy(ret.indices, stackIndices, sizeof(stackIndices));


	generateNormals(ret);
	ret.createVertexBuffer();
	return ret;
}

MeshData MeshGenerator::makeCylinder( const float height, const float radiusTop, const float radiusBottom,
	const QVector4D &colorTop, const QVector4D &colorBottom, const int segments)
{
	MeshData ret;

	int nbSegments;
	QVector<vec4> vertex_positions;
	QVector<vec4> vertex_colors;
	QVector<GLuint>	 vertex_indexes;

	// generate a cylinder with the bottom base center at (0,0,0), up on the Y axis
	vertex_positions.empty();
	vertex_colors.empty();
	nbSegments = segments;

	double angle = 0.0;
	vertex_positions.push_back(vec4(0,height,0,1));
	vertex_colors.push_back(colorTop);
	for(int i = 0; i<nbSegments; ++i)
	{
		angle = ((double)i)/((double)nbSegments)*2.0*3.14;
		vertex_positions.push_back(vec4(radiusTop*std::cos(angle),height,radiusTop*std::sin(angle),1.0));
		vertex_colors.push_back(colorTop);
		vertex_indexes.push_back(0);
		vertex_indexes.push_back((i+1)%nbSegments + 1);
		vertex_indexes.push_back(i+1);
	}

	vertex_positions.push_back(vec4(0,0,0,1));
	vertex_colors.push_back(colorBottom);
	for(int i = 0; i<nbSegments; ++i)
	{
		angle = ((double)i)/((double)nbSegments)*2.0*3.14;
		vertex_positions.push_back(vec4(radiusBottom*std::cos(angle),0.0,radiusBottom*std::sin(angle),1.0));
		vertex_colors.push_back(colorBottom);
		vertex_indexes.push_back(nbSegments+1);
		vertex_indexes.push_back(nbSegments+2+(i+1)%nbSegments);
		vertex_indexes.push_back(nbSegments+i+2);
	}

	for(int i = 0; i<nbSegments; ++i)
	{
		vertex_indexes.push_back(i+1);
		vertex_indexes.push_back((i+1)%nbSegments + 1);
		vertex_indexes.push_back(nbSegments+2+(i+1)%nbSegments);

		vertex_indexes.push_back(i+1);
		vertex_indexes.push_back(nbSegments+2+(i+1)%nbSegments);
		vertex_indexes.push_back(nbSegments+i+2);
	}

	// map the vertices into mesh data
	ret.numVertices = vertex_positions.size();
	ret.vertices = new Vertex[ret.numVertices];
	for (int i = 0; i < vertex_positions.size(); ++i)
	{
		ret.vertices[i].postition = vec3(vertex_positions[i].x(), vertex_positions[i].y(), vertex_positions[i].z());
		ret.vertices[i].color = vertex_colors[i];
	}
	
	ret.numIndices = vertex_indexes.size();
	ret.indices = new GLushort[ret.numIndices];
	for (int i = 0; i < vertex_indexes.size(); ++i)
	{
		ret.indices[i] = vertex_indexes[i];
	}

	generateNormals(ret);

	ret.createVertexBuffer();

	return ret;
}

MeshData MeshGenerator::makeCylinder( const float height, const float radiusTop, const float radiusBottom, const QColor &colorTop, const QColor &colorBottom, const int segments /*= 16*/ )
{
	vec4 top = vec4((GLfloat)colorTop.red()/255, (GLfloat)colorTop.green()/255, (GLfloat)colorTop.blue()/255, 1.0);
	vec4 bot = vec4((GLfloat)colorBottom.red()/255, (GLfloat)colorBottom.green()/255, (GLfloat)colorBottom.blue()/255, 1.0);
	return makeCylinder(height, radiusTop, radiusBottom, top, bot);
}



MeshData MeshGenerator::makeDemoRoom()
{
	MeshData ret;
	Vertex stackVerts[] = {
		vec3(-2.0f, +3.0f, 2.0f), // 0
		vec4(0.25f, 0.25f, 0.25f, 1.0f),
		vec3(1.0f, 0.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(-2.0f, +3.0f, -2.0f), // 1
		vec4(0.25f, 0.25f, 0.25f, 1.0f),
		vec3(1.0f, 0.0f, 1.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(+2.0f, +3.0f, -2.0f), // 2
		vec4(0.25f, 0.25f, 0.25f, 1.0f),
		vec3(-1.0f, -1.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(+2.0f, +3.0f, 2.0f), // 3
		vec4(0.25f, 0.25f, 0.25f, 1.0f),
		vec3(0.0f, 0.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(-2.0f, -0.1f, 2.0f), // 4
		vec4(0.25f, 0.25f, 0.25f, 1.0f),
		vec3(1.0f, 1.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(-2.0f, -0.1f, -2.0f), // 5
		vec4(0.25f, 0.25f, 0.25f, 1.0f),
		vec3(1.0f, 1.0f, 1.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(+2.0f, -0.1f, -2.0f), // 6
		vec4(0.25f, 0.25f, 0.25f, 1.0f),
		vec3(0.0f, 1.0f, 1.0f),
		vec2(0, 0),
		VertexBoneData(),

		vec3(+2.0f, -0.1f, 2.0f), // 7
		vec4(0.25f, 0.25f, 0.25f, 1.0f),
		vec3(0.0f, 1.0f, 0.0f),
		vec2(0, 0),
		VertexBoneData()

	};

	ret.numVertices = 8;
	ret.vertices = new Vertex[ret.numVertices];
	memcpy(ret.vertices, stackVerts, sizeof(stackVerts));

	unsigned short stackIndices[] = {
		0,4,5,5,1,0, // Left
		1,5,6,6,2,1, // Back
		5,4,7,7,6,5 // Bottom
	};

	ret.numIndices = NUM_ARRAY_ELEMENTS(stackIndices);
	ret.indices = new GLushort[ret.numIndices];
	memcpy(ret.indices, stackIndices, sizeof(stackIndices));

	//generateNormals(ret);
	return ret;
}

MeshData MeshGenerator::makePyramid()
{
	MeshData ret;

	return ret;
}

Bone* MeshGenerator::makeHand()
{
	MeshData mesh_root = makePalm(vec4(0.6, 0.6, 1.0, 1.0));

	MeshData mesh_1_1 = makeCylinder(0.028f, 0.0046f, 0.005f, QColor(0,153,255), QColor(0,153,255));
	MeshData mesh_1_2 = makeCylinder(0.023f, 0.0042f, 0.0046f, QColor(34,139,34), QColor(34,139,34));
	MeshData mesh_1_3 = makeCylinder(0.02f, 0.0038f, 0.0042f, QColor(255,12,62), QColor(255,12,62));

	MeshData mesh_2_1 = makeCylinder(0.035f, 0.0046f, 0.005f, QColor(0,153,255), QColor(0,153,255));
	MeshData mesh_2_2 = makeCylinder(0.025f, 0.0042f, 0.0046f, QColor(34,139,34), QColor(34,139,34));
	MeshData mesh_2_3 = makeCylinder(0.015f, 0.0038f, 0.0042f, QColor(255,12,62), QColor(255,12,62));

	MeshData mesh_3_1 = makeCylinder(0.04f, 0.0046f, 0.005f, QColor(0,153,255), QColor(0,153,255));
	MeshData mesh_3_2 = makeCylinder(0.03f, 0.0042f, 0.0046f, QColor(34,139,34), QColor(34,139,34));
	MeshData mesh_3_3 = makeCylinder(0.015f, 0.0038f, 0.0042f,QColor(255,12,62), QColor(255,12,62));

	MeshData mesh_4_1 = makeCylinder(0.035f, 0.0046f, 0.005f, QColor(0,153,255), QColor(0,153,255));
	MeshData mesh_4_2 = makeCylinder(0.03f, 0.0042f, 0.0046f, QColor(34,139,34), QColor(34,139,34));
	MeshData mesh_4_3 = makeCylinder(0.015f, 0.0038f, 0.0042f,QColor(255,12,62), QColor(255,12,62));

	MeshData mesh_5_1 = makeCylinder(0.02f, 0.0046f, 0.005f, QColor(0,153,255), QColor(0,153,255));
	MeshData mesh_5_2 = makeCylinder(0.02f, 0.0042f, 0.0046f, QColor(34,139,34), QColor(34,139,34));
	MeshData mesh_5_3 = makeCylinder(0.01f, 0.0038f, 0.0042f, QColor(255,12,62), QColor(255,12,62));

	mat4 tMatrix;

	
	// root
	Bone* root = new Bone("root", NULL, tMatrix, mesh_root);
	
	// thumb
	tMatrix.setToIdentity();
	tMatrix.translate(0.03, 0.01, -0.01);
	tMatrix.rotate(80, vec3(0, 1, 0));
	tMatrix.rotate(-60, vec3(0, 0, 1));
	tMatrix.rotate(60, vec3(1, 0, 0));
	
	Bone* thumb_1 = new Bone("thumb_1", root, tMatrix, mesh_1_1);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.028, 0);
	Bone* thumb_2 = new Bone("thumb_2", thumb_1, tMatrix, mesh_1_2);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.023, 0);
	Bone* thumb_3 = new Bone("thumb_3", thumb_2, tMatrix, mesh_1_3);

	// index finger
	tMatrix.setToIdentity();
	tMatrix.translate(0.024, 0.05, 0);
	Bone* index_1 = new Bone("index_1", root, tMatrix, mesh_2_1);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.035, 0);
	Bone* index_2 = new Bone("index_2", index_1, tMatrix, mesh_2_2);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.025, 0);
	Bone* index_3 = new Bone("index_3", index_2, tMatrix, mesh_2_3);

	// middle finger
	tMatrix.setToIdentity();
	tMatrix.translate(0.008, 0.05, 0);
	Bone* middle_1 = new Bone("middle_1", root, tMatrix, mesh_3_1);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.04, 0);
	Bone* middle_2 = new Bone("middle_2", middle_1, tMatrix, mesh_3_2);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.03, 0);
	Bone* middle_3 = new Bone("middle_3", middle_2, tMatrix, mesh_3_3);

	// ring finger
	tMatrix.setToIdentity();
	tMatrix.translate(-0.008, 0.05, 0);
	Bone* ring_1 = new Bone("ring_1", root, tMatrix, mesh_4_1);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.035, 0);
	Bone* ring_2 = new Bone("ring_2", ring_1, tMatrix, mesh_4_2);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.03, 0);
	Bone* ring_3 = new Bone("ring_3", ring_2, tMatrix, mesh_4_3);

	// little finger
	tMatrix.setToIdentity();
	tMatrix.translate(-0.024, 0.05, 0);
	Bone* little_1 = new Bone("little_1", root, tMatrix, mesh_5_1);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.02, 0);
	Bone* little_2 = new Bone("little_2", little_1, tMatrix, mesh_5_2);

	tMatrix.setToIdentity();
	tMatrix.translate(0, 0.02, 0);
	Bone* little_3 = new Bone("little_3", little_2, tMatrix, mesh_5_3);

	return root;
}


