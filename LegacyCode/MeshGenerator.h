#pragma once
#include <Primitives/MeshData.h>
#include <Primitives/Bone.h>

class MeshGenerator
{
public:
	static MyMeshData makeTriangle();
	static MyMeshData makeIdentityCube(vec4 &color);
	static MyMeshData makeCylinder(const float height, const float radiusTop, const float radiusBottom,
		const QVector4D &colorTop, const QVector4D &colorBottom, const int segments = 16);
	static MyMeshData makeCylinder(const float height, const float radiusTop, const float radiusBottom,
		const QColor &colorTop, const QColor &colorBottom, const int segments = 16);
	static MyMeshData makeDemoRoom();
	static MyMeshData makePyramid();
	static MyMeshData makePalm(vec4 &color);
	static Bone* makeHand(); // returns the root of the hand


	static void generateNormals(MyMeshData &mesh);

};

