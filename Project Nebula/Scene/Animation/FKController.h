#pragma once
#include <Utility/ModelLoader.h>
#include <assimp/scene.h>

class FKController
{
public:
	FKController(ModelLoader* loader);
	~FKController(void);

	void BoneTransform(float TimeInSeconds, QVector<mat4>& Transforms);

private:
	void calcInterpolatedScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void calcInterpolatedRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);
	void calcInterpolatedPosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim);    
	uint findScaling(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint findRotation(float AnimationTime, const aiNodeAnim* pNodeAnim);
	uint findPosition(float AnimationTime, const aiNodeAnim* pNodeAnim);
	const aiNodeAnim* findNodeAnim(const aiAnimation* pAnimation, const QString NodeName);
	void readNodeHeirarchy(float AnimationTime, const aiNode* pNode, const mat4& ParentTransform);

	QMap<QString, uint> m_BoneMapping; // maps a bone name to its index
	mat4 m_GlobalInverseTransform;
	uint m_NumBones;
	QVector<ModelLoader::BoneInfo> m_BoneInfo;
	aiAnimation** m_Animations;
	aiNode* m_root;
};

