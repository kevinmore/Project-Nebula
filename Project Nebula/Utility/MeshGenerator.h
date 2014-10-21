#pragma once
#include <Primitives/MeshData.h>
#include <Primitives/Bone.h>

class MeshGenerator
{
public:
	static MeshData makeTriangle();
	static MeshData makeIdentityCube(vec4 &color);
	static MeshData makeCylinder(const float height, const float radiusTop, const float radiusBottom,
		const QVector4D &colorTop, const QVector4D &colorBottom, const int segments = 16);
	static MeshData makeCylinder(const float height, const float radiusTop, const float radiusBottom,
		const QColor &colorTop, const QColor &colorBottom, const int segments = 16);
	static MeshData makeDemoRoom();
	static MeshData makePyramid();
	static MeshData makePalm(vec4 &color);
	static Bone* makeHand(); // returns the root of the hand


	static void generateNormals(MeshData &mesh);

};

