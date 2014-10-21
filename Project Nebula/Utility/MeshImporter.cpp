#include <Utility/MeshImporter.h>
#include <assert.h>


// utility function to convert aiMatrix4x4 to QMatrix4x4
QMatrix4x4 convToQMat4(const aiMatrix4x4 * m)
{
	return QMatrix4x4(m->a1, m->b1, m->c1, m->d1,
		m->a2, m->b2, m->c2, m->d2,
		m->a3, m->b3, m->c3, m->d3,
		m->a4, m->b4, m->c4, m->d4);
}

QMatrix4x4 convToQMat4(aiMatrix4x4 * m) 
{
	return QMatrix4x4(m->a1, m->b1, m->c1, m->d1,
		m->a2, m->b2, m->c2, m->d2,
		m->a3, m->b3, m->c3, m->d3,
		m->a4, m->b4, m->c4, m->d4);
}

QMatrix4x4 convToQMat4(aiMatrix3x3 * m) 
{
	return QMatrix4x4(m->a1, m->b1, m->c1, 0,
						m->a2, m->b2, m->c2, 0,
						m->a3, m->b3, m->c3, 0,
						0, 0, 0, 1);
}


MeshImporter::MeshImporter(void)
{
	cleanUp();
	m_NumBones = 0;
	m_Root = new Bone();
}


MeshImporter::~MeshImporter(void)
{
}


void MeshImporter::cleanUp()
{
	m_Materials.clear();
	m_Meshes.clear();
	m_BoneMapping.clear();
	m_Bones.clear();
	m_Entries.clear();
	m_BoneInfo.clear();
}


bool MeshImporter::loadMeshFromFile( const QString &fileName )
{
	Assimp::Importer* importer = new Assimp::Importer();

	    m_pScene = importer->ReadFile(fileName.toStdString(),
		aiProcess_GenSmoothNormals |
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType |
		aiProcess_FlipUVs
		);

	if (!m_pScene)
	{
		qDebug() << "Error loading mesh file: " << fileName << importer->GetErrorString();
		return false;
	}
	else
	{
		m_GlobalInverseTransform = convToQMat4(&m_pScene->mRootNode->mTransformation);
		m_GlobalInverseTransform.inverted();
		processScene(m_pScene, fileName);
	}

	return true;

}

bool MeshImporter::processScene( const aiScene* pScene, const QString &fileName )
{
	m_Entries.resize(pScene->mNumMeshes);

	uint NumVertices = 0;
	uint NumIndices = 0;

	// Count the number of vertices and indices
	for (int i = 0 ; i < m_Entries.size() ; ++i) 
	{
		m_Entries[i].MaterialIndex = pScene->mMeshes[i]->mMaterialIndex;        
		m_Entries[i].NumIndices    = pScene->mMeshes[i]->mNumFaces * 3;
		m_Entries[i].BaseVertex    = NumVertices;
		m_Entries[i].BaseIndex     = NumIndices;

		NumVertices += pScene->mMeshes[i]->mNumVertices;
		NumIndices  += m_Entries[i].NumIndices;
	}

	// load materials
	if (pScene->HasMaterials())
	{
		m_Materials.reserve(pScene->mNumMaterials);
		for (unsigned int i = 0; i < pScene->mNumMaterials; ++i)
		{
			m_Materials.push_back(processMaterial(pScene->mMaterials[i], fileName));
		}
	}

	// load mesh
	if (pScene->HasMeshes())
	{
		for (int i = 0; i < m_Entries.size(); ++i)
		{
			m_Meshes.push_back(processMesh(i, pScene->mMeshes[i]));
		}
	}
	else
	{
		qDebug() << "Error: No meshes found" << fileName;
		return false;
	}

	// load bones
	if (pScene->mRootNode != NULL)
	{
		//processSkeleton(pScene, pScene->mRootNode, 0, *m_Root);
		
	}
	else
	{
		qDebug() << "Error: The model has no root node" << fileName;
		return false;
	}

	// load animation
	if (pScene->HasAnimations())
	{
		for (uint i = 0; i < pScene->mNumAnimations; ++i)
		{
			m_Animations.push_back(processAnimations(i));
		}
		
	}
	else
	{
		qDebug() << "The model has no animations.";
	}
	return true;
}

MaterialInfo* MeshImporter::processMaterial( const aiMaterial *pMaterial, const QString &fileName )
{
	MaterialInfo* mater = new MaterialInfo();

	// Extract the directory part from the file name
	int SlashIndex = fileName.lastIndexOf("/");
	QString Dir;

	if (SlashIndex == std::string::npos) {
		Dir = ".";
	}
	else if (SlashIndex == 0) {
		Dir = "/";
	}
	else {
		QString left = fileName.left(SlashIndex);
		Dir = left;
	}

	// Initialize the materials
	if (pMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0) 
	{
		aiString textureFileName;
		if (pMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &textureFileName, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS)
		{
			QString FullPath = Dir + "/" + textureFileName.data;
			mater->textureFile = new Texture(FullPath);
		}
		else
		{
			qDebug() << "This mesh has no textures.";
		}
	}

	int shadingModel;
	pMaterial->Get(AI_MATKEY_SHADING_MODEL, shadingModel);

	if (shadingModel != aiShadingMode_Phong && shadingModel != aiShadingMode_Gouraud)
	{
		mater->Name = "DefaultMaterial";
	}
	else
	{
		aiColor3D dif(0.f,0.f,0.f);
		aiColor3D amb(0.f,0.f,0.f);
		aiColor3D spec(0.f,0.f,0.f);
		float shine = 0.0;

		pMaterial->Get(AI_MATKEY_COLOR_AMBIENT, amb);
		pMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, dif);
		pMaterial->Get(AI_MATKEY_COLOR_SPECULAR, spec);
		pMaterial->Get(AI_MATKEY_SHININESS, shine);

		mater->Ambient = QVector3D(amb.r, amb.g, amb.b);
		mater->Diffuse = QVector3D(dif.r, dif.g, dif.b);
		mater->Specular = QVector3D(spec.r, spec.g, spec.b);
		mater->Shininess = shine;

		mater->Ambient *= .2;
		if (mater->Shininess == 0.0)
			mater->Shininess = 30;
	}

	return mater;
}

MeshData* MeshImporter::processMesh(uint MeshIndex, const aiMesh* paiMesh)
{
	// process bones first
	QVector<VertexBoneData> Bones;
	Bones.resize(paiMesh->mNumVertices);
	if(paiMesh->HasBones()) processBones(MeshIndex, paiMesh, Bones);
	else qDebug() << "This mesh has no bones.";

	// Map those vectors into the my mesh data
	MeshData *ret(new MeshData);
	ret->meshName = paiMesh->mName.length != 0 ? paiMesh->mName.C_Str() : "";

	// Get vertex attributes
	ret->numVertices = paiMesh->mNumVertices;
	ret->vertices = new Vertex[ret->numVertices];

	for (uint i = 0 ; i < paiMesh->mNumVertices ; ++i) 
	{
		// position
		ret->vertices[i].postition = vec3 (paiMesh->mVertices[i].x, paiMesh->mVertices[i].y, paiMesh->mVertices[i].z);

		// Color
// 		if (paiMesh->HasVertexColors(1))
// 		{
// 			ret->vertices[i].color = vec4(paiMesh->mColors[i]->r, paiMesh->mColors[i]->g, paiMesh->mColors[i]->b, paiMesh->mColors[i]->a);
// 		}
// 		else
		{
			ret->vertices[i].color = vec4(0.547, 0.39, 0.234, 1);
		}

		// normals
		if (paiMesh->HasNormals())
		{
			ret->vertices[i].normal = vec3 ( paiMesh->mNormals[i].x, paiMesh->mNormals[i].y, paiMesh->mNormals[i].z);
		}

		// texture coord
		if (paiMesh->HasTextureCoords(0))
		{
			ret->vertices[i].texCoord = vec2 (paiMesh->mTextureCoords[0][i].x,  paiMesh->mTextureCoords[0][i].y);
		}

		// bone data
		ret->vertices[i].boneIDs = vec4(Bones[i].IDs[0], Bones[i].IDs[1], Bones[i].IDs[2], Bones[i].IDs[3]);
		ret->vertices[i].boneWeights = vec4(Bones[i].Weights[0], Bones[i].Weights[1], Bones[i].Weights[2], Bones[i].Weights[3]);
	}


	// Get mesh indexes
	ret->numIndices = 3 * paiMesh->mNumFaces;
	ret->indices = new GLushort[ret->numIndices];
	for (uint i = 0; i < paiMesh->mNumFaces; ++i)
	{
		const aiFace& face = paiMesh->mFaces[i];
		if (face.mNumIndices != 3)
		{
			qDebug() << "Warning: Mesh face with not exactly 3 indices, ignoring this primitive.";
			continue;
		}
		ret->indices[3*i] = face.mIndices[0];
		ret->indices[3*i+1] = face.mIndices[1];
		ret->indices[3*i+2] = face.mIndices[2];
	}

	// material
	ret->material = m_Materials[paiMesh->mMaterialIndex];


	// create a vertex buffer
	ret->createVertexBuffer();

	return ret;
}

void MeshImporter::processBones( uint MeshIndex, const aiMesh *paiMesh, QVector<VertexBoneData> &Bones )
{
	for (uint i = 0; i < paiMesh->mNumBones; ++i)
	{
		uint boneIndex = 0;        
		QString boneName(paiMesh->mBones[i]->mName.data);
		qDebug() << "Bone detected:" << boneName;
		if (m_BoneMapping.find(boneName) == m_BoneMapping.end()) 
		{
			// Allocate an index for a new bone
			boneIndex = m_NumBones;
			m_NumBones++;
			BoneInfo bi;			
			m_BoneInfo.push_back(bi);
			m_BoneInfo[boneIndex].boneOffset = convToQMat4(&paiMesh->mBones[i]->mOffsetMatrix);
			m_BoneMapping[boneName] = boneIndex;

			m_BoneMapping[boneName] = boneIndex;
		}
		else 
		{
			boneIndex = m_BoneMapping[boneName];
		}                      

		for (uint j = 0 ; j < paiMesh->mBones[i]->mNumWeights ; ++j) 
		{
			uint VertexID = paiMesh->mBones[i]->mWeights[j].mVertexId;
			float Weight  = paiMesh->mBones[i]->mWeights[j].mWeight;                   
			Bones[VertexID].addBoneData(boneIndex, Weight);
		}
	}
}

aiAnimation* MeshImporter::processAnimations( uint animationIndex )
{
	return m_pScene->mAnimations[animationIndex];
}

void MeshImporter::processSkeleton( const aiScene *scene, aiNode *node, Bone *parent, Bone &currentBone )
{
	currentBone.m_boneName = node->mName.length != 0 ? node->mName.C_Str() : "";

	currentBone.m_localTransform = convToQMat4(&node->mTransformation);

	if(parent) parent->addChild(&currentBone);

	if (m_BoneMapping.find(currentBone.m_boneName) != m_BoneMapping.end()) {
		qDebug() << "Bone matches: " << currentBone.m_boneName;
	}

	for (uint imesh = 0; imesh < node->mNumMeshes; ++imesh)
	{
		MeshData* mesh = m_Meshes[node->mMeshes[imesh]];
		currentBone.m_Meshes.push_back(*mesh);
	}

	for (uint ich = 0; ich < node->mNumChildren; ++ich)
	{
		Bone* temp = new Bone();
		currentBone.addChild(temp);
		processSkeleton(scene, node->mChildren[ich], &currentBone, *currentBone.m_children[ich]);
		temp = 0;
		delete temp;
	}
}

void MeshImporter::ReadNodeHeirarchy( float AnimationTime, const aiNode* pNode, const mat4 &ParentTransform )
{
	QString NodeName(pNode->mName.data);

	const aiAnimation* pAnimation = m_pScene->mAnimations[0];

	mat4 NodeTransformation(convToQMat4(&pNode->mTransformation));

	const aiNodeAnim* pNodeAnim = FindNodeAnim(pAnimation, NodeName);

	if (pNodeAnim) {
		// Interpolate scaling and generate scaling transformation matrix
		aiVector3D Scaling;
		CalcInterpolatedScaling(Scaling, AnimationTime, pNodeAnim);
		mat4 ScalingM;
		ScalingM.scale(Scaling.x, Scaling.y, Scaling.z);

		// Interpolate rotation and generate rotation transformation matrix
		aiQuaternion RotationQ;
		CalcInterpolatedRotation(RotationQ, AnimationTime, pNodeAnim);        
		mat4 RotationM = convToQMat4(&RotationQ.GetMatrix());

		// Interpolate translation and generate translation transformation matrix
		aiVector3D Translation;
		CalcInterpolatedPosition(Translation, AnimationTime, pNodeAnim);
		mat4 TranslationM;
		TranslationM.translate(Translation.x, Translation.y, Translation.z);

		// Combine the above transformations
		NodeTransformation = TranslationM * RotationM * ScalingM;
	}

	mat4 GlobalTransformation = ParentTransform * NodeTransformation;

	if (m_BoneMapping.find(NodeName) != m_BoneMapping.end()) {
		uint BoneIndex = m_BoneMapping[NodeName];
		m_BoneInfo[BoneIndex].finalTransformation = m_GlobalInverseTransform * GlobalTransformation * m_BoneInfo[BoneIndex].boneOffset;
	}

	for (uint i = 0 ; i < pNode->mNumChildren ; i++) {
		ReadNodeHeirarchy(AnimationTime, pNode->mChildren[i], GlobalTransformation);
	}

}

const aiNodeAnim* MeshImporter::FindNodeAnim( const aiAnimation* pAnimation, QString &NodeName )
{
	for (uint i = 0 ; i < pAnimation->mNumChannels ; i++) {
		const aiNodeAnim* pNodeAnim = pAnimation->mChannels[i];

		if (QString(pNodeAnim->mNodeName.data) == NodeName) {
			return pNodeAnim;
		}
	}

	return NULL;
}


void MeshImporter::CalcInterpolatedPosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	if (pNodeAnim->mNumPositionKeys == 1) {
		Out = pNodeAnim->mPositionKeys[0].mValue;
		return;
	}

	uint PositionIndex = FindPosition(AnimationTime, pNodeAnim);
	uint NextPositionIndex = (PositionIndex + 1);
	assert(NextPositionIndex < pNodeAnim->mNumPositionKeys);
	float DeltaTime = (float)(pNodeAnim->mPositionKeys[NextPositionIndex].mTime - pNodeAnim->mPositionKeys[PositionIndex].mTime);
	float Factor = (AnimationTime - (float)pNodeAnim->mPositionKeys[PositionIndex].mTime) / DeltaTime;
	assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiVector3D& Start = pNodeAnim->mPositionKeys[PositionIndex].mValue;
	const aiVector3D& End = pNodeAnim->mPositionKeys[NextPositionIndex].mValue;
	aiVector3D Delta = End - Start;
	Out = Start + Factor * Delta;
}


void MeshImporter::CalcInterpolatedRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	// we need at least two values to interpolate...
	if (pNodeAnim->mNumRotationKeys == 1) {
		Out = pNodeAnim->mRotationKeys[0].mValue;
		return;
	}

	uint RotationIndex = FindRotation(AnimationTime, pNodeAnim);
	uint NextRotationIndex = (RotationIndex + 1);
	assert(NextRotationIndex < pNodeAnim->mNumRotationKeys);
	float DeltaTime = (float)(pNodeAnim->mRotationKeys[NextRotationIndex].mTime - pNodeAnim->mRotationKeys[RotationIndex].mTime);
	float Factor = (AnimationTime - (float)pNodeAnim->mRotationKeys[RotationIndex].mTime) / DeltaTime;
	assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiQuaternion& StartRotationQ = pNodeAnim->mRotationKeys[RotationIndex].mValue;
	const aiQuaternion& EndRotationQ   = pNodeAnim->mRotationKeys[NextRotationIndex].mValue;    
	aiQuaternion::Interpolate(Out, StartRotationQ, EndRotationQ, Factor);
	Out = Out.Normalize();
}


void MeshImporter::CalcInterpolatedScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	if (pNodeAnim->mNumScalingKeys == 1) {
		Out = pNodeAnim->mScalingKeys[0].mValue;
		return;
	}

	uint ScalingIndex = FindScaling(AnimationTime, pNodeAnim);
	uint NextScalingIndex = (ScalingIndex + 1);
	assert(NextScalingIndex < pNodeAnim->mNumScalingKeys);
	float DeltaTime = (float)(pNodeAnim->mScalingKeys[NextScalingIndex].mTime - pNodeAnim->mScalingKeys[ScalingIndex].mTime);
	float Factor = (AnimationTime - (float)pNodeAnim->mScalingKeys[ScalingIndex].mTime) / DeltaTime;
	assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiVector3D& Start = pNodeAnim->mScalingKeys[ScalingIndex].mValue;
	const aiVector3D& End   = pNodeAnim->mScalingKeys[NextScalingIndex].mValue;
	aiVector3D Delta = End - Start;
	Out = Start + Factor * Delta;
}

uint MeshImporter::FindPosition(float AnimationTime, const aiNodeAnim* pNodeAnim)
{    
	for (uint i = 0 ; i < pNodeAnim->mNumPositionKeys - 1 ; i++) {
		if (AnimationTime < (float)pNodeAnim->mPositionKeys[i + 1].mTime) {
			return i;
		}
	}

	assert(0);

	return 0;
}


uint MeshImporter::FindRotation(float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	assert(pNodeAnim->mNumRotationKeys > 0);

	for (uint i = 0 ; i < pNodeAnim->mNumRotationKeys - 1 ; i++) {
		if (AnimationTime < (float)pNodeAnim->mRotationKeys[i + 1].mTime) {
			return i;
		}
	}

	assert(0);

	return 0;
}


uint MeshImporter::FindScaling(float AnimationTime, const aiNodeAnim* pNodeAnim)
{
	assert(pNodeAnim->mNumScalingKeys > 0);

	for (uint i = 0 ; i < pNodeAnim->mNumScalingKeys - 1 ; i++) {
		if (AnimationTime < (float)pNodeAnim->mScalingKeys[i + 1].mTime) {
			return i;
		}
	}

	assert(0);

	return 0;
}

void MeshImporter::BoneTransform( float TimeInSeconds, QVector<mat4> &Transforms )
{
	if(m_Animations.isEmpty()) return;

	mat4 Identity;
	Identity.setToIdentity();

	float TicksPerSecond = (float)(m_Animations[0]->mTicksPerSecond != 0 ? m_Animations[0]->mTicksPerSecond : 25.0f);
	float TimeInTicks = TimeInSeconds * TicksPerSecond;
	float AnimationTime = fmod(TimeInTicks, (float)m_Animations[0]->mDuration);

	ReadNodeHeirarchy(AnimationTime, m_pScene->mRootNode, Identity);

	Transforms.resize(m_NumBones);

	for (uint i = 0 ; i < m_NumBones ; i++) {
		Transforms[i] = m_BoneInfo[i].finalTransformation;
	}
}
