#include <Utility/MeshImporter.h>
#include <assert.h>
#define POSITION_LOCATION    0
#define TEX_COORD_LOCATION   1
#define NORMAL_LOCATION      2
#define BONE_ID_LOCATION     3
#define BONE_WEIGHT_LOCATION 4

// utility function to convert aiMatrix4x4 to QMatrix4x4
QMatrix4x4 convToQMat4(const aiMatrix4x4 * m)
{
	return QMatrix4x4(m->a1, m->a2, m->a3, m->a4,
		m->b1, m->b2, m->b3, m->b4,
		m->c1, m->c2, m->c3, m->c4,
		m->d1, m->d2, m->d3, m->d4);
}

QMatrix4x4 convToQMat4(aiMatrix4x4 * m) 
{
	return QMatrix4x4(m->a1, m->a2, m->a3, m->a4,
		m->b1, m->b2, m->b3, m->b4,
		m->c1, m->c2, m->c3, m->c4,
		m->d1, m->d2, m->d3, m->d4);
}

QMatrix4x4 convToQMat4(aiMatrix3x3 * m) 
{
	return QMatrix4x4(m->a1, m->a2, m->a3, 0,
		m->b1, m->b2, m->b3, 0,
		m->c1, m->c2, m->c3, 0,
		0, 0, 0, 1);
}


MeshImporter::MeshImporter(void)
{
	
	m_NumBones = 0;
	m_offSet = 0;
	m_pScene = NULL;
	m_wholeMesh = NULL;
	m_loaded = false;
	m_VAO = 0;
	ZERO_MEM(m_Buffers);
	
}


MeshImporter::~MeshImporter(void)
{
	cleanUp();
}


void MeshImporter::cleanUp()
{
	if (m_Buffers[0] != 0) {
		glDeleteBuffers(ARRAY_SIZE_IN_ELEMENTS(m_Buffers), m_Buffers);
	}

	if (m_VAO != 0) {
		glDeleteVertexArrays(1, &m_VAO);
		m_VAO = 0;
	}
	m_Materials.clear();
	m_Meshes.clear();
	m_BoneMapping.clear();
	m_Entries.clear();
	m_BoneInfo.clear();	
}


bool MeshImporter::loadMeshFromFile( const QString &fileName )
{
	 cleanUp();
	// Create the VAO
	 glewExperimental = GL_TRUE;
	 glewInit();
	glGenVertexArrays(1, &m_VAO);   
	glBindVertexArray(m_VAO);

	// Create the buffers for the vertices attributes
	glGenBuffers(ARRAY_SIZE_IN_ELEMENTS(m_Buffers), m_Buffers);

	m_pScene = importer.ReadFile(fileName.toStdString(),
	aiProcess_GenSmoothNormals |
//	aiProcess_CalcTangentSpace |
	aiProcess_Triangulate |
//	aiProcess_JoinIdenticalVertices |
	aiProcess_SortByPType |
	aiProcess_FlipUVs
	);

	if (!m_pScene)
	{
		qDebug() << "Error loading mesh file: " << fileName << importer.GetErrorString();
		m_loaded = false;
	}
	else
	{
 		m_GlobalInverseTransform = convToQMat4(&m_pScene->mRootNode->mTransformation);
 		m_GlobalInverseTransform.inverted();
// 		m_GlobalInverseTransform = mat4(1, 0, 0, 0, 
// 										0, 0, -1, 0,
// 										0, 1, 0, 0,
// 										0, 0, 0, 1);
		processScene(m_pScene, fileName);
		m_loaded = true;
	}

	// Make sure the VAO is not changed from the outside
	glBindVertexArray(0);	

	return m_loaded;

}

bool MeshImporter::processScene( const aiScene* pScene, const QString &fileName )
{
	m_Entries.resize(pScene->mNumMeshes);

	uint NumVertices = 0;
	uint NumIndices = 0;

	// Count the number of vertices and indices
	for (uint i = 0 ; i < pScene->mNumMeshes; i++) {
		m_Entries[i].MaterialIndex = pScene->mMeshes[i]->mMaterialIndex;        
		m_Entries[i].NumIndices    = pScene->mMeshes[i]->mNumFaces * 3;
		m_Entries[i].BaseVertex    = NumVertices;
		m_Entries[i].BaseIndex     = NumIndices;

		NumVertices += pScene->mMeshes[i]->mNumVertices;
		NumIndices  += m_Entries[i].NumIndices;
	}



	// Reserve space in the vectors for the vertex attributes and indices
	QVector<vec3> Positions;
	QVector<vec3> Normals;
	QVector<vec2> TexCoords;
	QVector<VertexBoneData> Bones;
	QVector<uint> Indices;

	Positions.reserve(NumVertices);
	Normals.reserve(NumVertices);
	TexCoords.reserve(NumVertices);
	Bones.resize(NumVertices);
	Indices.reserve(NumIndices);


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
		for (uint i = 0; i < pScene->mNumMeshes; ++i)
		{
			const aiMesh* paiMesh = pScene->mMeshes[i];
			processMesh(i, paiMesh, Positions, Normals, TexCoords, Bones, Indices);
		}

// 		for (int i = 0; i < m_Meshes.size(); ++i)
// 		{
// 			MeshData* pMesh = m_Meshes[i];
// 			for (int j = 0; j < pMesh->numVertices; ++j)
// 			{
// 				m_wholeMesh->vertices[m_Entries[i].BaseVertex + j] = pMesh->vertices[j];
// 			}
// 			for (int j = 0; j < pMesh->numIndices; ++j)
// 			{
// 				m_wholeMesh->indices[m_Entries[i].BaseIndex + j] = pMesh->indices[j] + m_Entries[i].BaseVertex;
// 			}
// 		}
// 		for (int i = 0; i < m_Textures.size(); ++i)
// 			m_wholeMesh->material->textures << m_Textures[i];
// 
// 		m_wholeMesh->createVertexBuffer();
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


	// Generate and populate the buffers with vertex attributes and the indices
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[POS_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Positions[0]) * Positions.size(), &Positions[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(POSITION_LOCATION);
	glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);    

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TEXCOORD_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(TexCoords[0]) * TexCoords.size(), &TexCoords[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(TEX_COORD_LOCATION);
	glVertexAttribPointer(TEX_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[NORMAL_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Normals[0]) * Normals.size(), &Normals[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(NORMAL_LOCATION);
	glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[BONE_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Bones[0]) * Bones.size(), &Bones[0], GL_STATIC_DRAW);
	glEnableVertexAttribArray(BONE_ID_LOCATION);
	glVertexAttribIPointer(BONE_ID_LOCATION, 4, GL_INT, sizeof(VertexBoneData), (const GLvoid*)0);
	glEnableVertexAttribArray(BONE_WEIGHT_LOCATION);    
	glVertexAttribPointer(BONE_WEIGHT_LOCATION, 4, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*)16);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Buffers[INDEX_BUFFER]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Indices[0]) * Indices.size(), &Indices[0], GL_STATIC_DRAW);


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
			Texture* pTex = new Texture(FullPath);
			mater->textures.push_back(pTex);
			m_Textures.push_back(pTex);
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

void MeshImporter::processMesh(uint MeshIndex,
							const aiMesh* paiMesh,
							QVector<vec3>& Positions,
							QVector<vec3>& Normals,
							QVector<vec2>& TexCoords,
							QVector<VertexBoneData>& Bones,
							QVector<unsigned int>& Indices)
{
	const aiVector3D Zero3D(0.0f, 0.0f, 0.0f);

	// Populate the vertex attribute vectors
	for (uint i = 0 ; i < paiMesh->mNumVertices ; i++) {
		const aiVector3D* pPos      = &(paiMesh->mVertices[i]);
		const aiVector3D* pNormal   = &(paiMesh->mNormals[i]);
		const aiVector3D* pTexCoord = paiMesh->HasTextureCoords(0) ? &(paiMesh->mTextureCoords[0][i]) : &Zero3D;

		Positions.push_back(vec3(pPos->x, pPos->y, pPos->z));
		Normals.push_back(vec3(pNormal->x, pNormal->y, pNormal->z));
		TexCoords.push_back(vec2(pTexCoord->x, pTexCoord->y));        
	}
	// process bones
	if(paiMesh->HasBones()) processBones(MeshIndex, paiMesh, Bones);
	else qDebug() << "This mesh" << paiMesh->mName.C_Str() << "has no bones.";

	// Populate the index buffer
	for (uint i = 0 ; i < paiMesh->mNumFaces ; i++) {
		const aiFace& Face = paiMesh->mFaces[i];
		if (Face.mNumIndices != 3)
		{
			qDebug() << "Warning: Mesh face with not exactly 3 indices, ignoring this primitive.";
			continue;
		}
		Indices.push_back(Face.mIndices[0]);
		Indices.push_back(Face.mIndices[1]);
		Indices.push_back(Face.mIndices[2]);
	}

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
		}
		else 
		{
			boneIndex = m_BoneMapping[boneName];
		}                      


		for (uint j = 0 ; j < paiMesh->mBones[i]->mNumWeights ; ++j) 
		{
			uint VertexID = m_Entries[MeshIndex].BaseVertex + paiMesh->mBones[i]->mWeights[j].mVertexId;
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

void MeshImporter::Render()
{
	glBindVertexArray(m_VAO);

	for (uint i = 0 ; i < m_Entries.size() ; i++) {
		const uint MaterialIndex = m_Entries[i].MaterialIndex;

		assert(MaterialIndex < m_Textures.size());

		if (m_Textures[MaterialIndex]) {
			m_Textures[MaterialIndex]->bind(GL_TEXTURE0);
		}

		glDrawElementsBaseVertex(GL_TRIANGLES, 
			m_Entries[i].NumIndices, 
			GL_UNSIGNED_INT, 
			(void*)(sizeof(uint) * m_Entries[i].BaseIndex), 
			m_Entries[i].BaseVertex);
	}

	// Make sure the VAO is not changed from the outside    
	glBindVertexArray(0);
}
