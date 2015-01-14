#include "ModelLoader.h"
#include <QDebug>
#include <Utility/Math.h>
#include <Animation/IK/FABRIKSolver.h>

ModelLoader::ModelLoader()
{
	initializeOpenGLFunctions();
}


void ModelLoader::clear()
{
	m_positions.clear();
	m_colors.clear();
	m_texCoords.clear();
	m_normals.clear();
	m_tangents.clear();
	m_indices.clear();
	m_Bones.clear();

	if (m_Buffers.size()) 
		glDeleteBuffers(m_Buffers.size(), m_Buffers.data());

	m_modelFeatures.hasColorMap = false;
	m_modelFeatures.hasNormalMap = false;
}

ModelLoader::~ModelLoader()
{
	clear();
	if (m_VAO != 0) 
	{
		glDeleteVertexArrays(1, &m_VAO);
		m_VAO = 0;
	}
}

QVector<ModelDataPtr> ModelLoader::loadModel( const QString& fileName )
{
	clear();

	m_scene = m_importer.ReadFile(fileName.toStdString(), aiProcessPreset_TargetRealtime_Quality | aiProcess_FlipUVs);

	if(!m_scene)
	{
		qDebug() << m_importer.GetErrorString();
		QVector<ModelDataPtr> empty;
		return empty;
	}
	else if(m_scene->HasTextures())
	{
		qFatal("Support for meshes with embedded textures is not implemented");
	}
	
   	m_GlobalInverseTransform = Math::convToQMat4(m_scene->mRootNode->mTransformation.Inverse());

	m_modelType = m_scene->HasAnimations() ? RIGGED_MODEL : STATIC_MODEL;

	unsigned int numVertices = 0;
	unsigned int numIndices  = 0;

	for(uint i = 0; i < m_scene->mNumMeshes; ++i)
	{
		numVertices += m_scene->mMeshes[i]->mNumVertices;
		numIndices  += m_scene->mMeshes[i]->mNumFaces * 3;
	}

	m_positions.reserve(numVertices);
	m_colors.reserve(numVertices);
	m_normals.reserve(numVertices);
	m_texCoords.reserve(numVertices);
	m_tangents.reserve(numVertices);
	m_indices.reserve(numIndices);
	if(m_modelType == RIGGED_MODEL)
		m_Bones.resize(numVertices);

	numVertices = 0;
	numIndices  = 0;
	m_NumBones = 0;

	QVector<ModelDataPtr> modelDataVector;
	modelDataVector.resize(m_scene->mNumMeshes);

	for(uint i = 0; i < m_scene->mNumMeshes; ++i)
	{
		ModelData* md = new ModelData();
		
		md->meshData     = loadMesh(i, numVertices, numIndices, m_scene->mMeshes[i], fileName);
		md->textureData  = loadTexture(fileName, m_scene->mMaterials[m_scene->mMeshes[i]->mMaterialIndex]);
		md->materialData = loadMaterial(i, m_scene->mMaterials[m_scene->mMeshes[i]->mMaterialIndex]);
		md->hasAnimation = m_scene->HasAnimations();

		// calculate the animation duration in seconds
		if(m_scene->HasAnimations()) md->animationDuration = (float) m_scene->mAnimations[0]->mDuration;

		numVertices += m_scene->mMeshes[i]->mNumVertices;
		numIndices  += m_scene->mMeshes[i]->mNumFaces * 3;
		
		modelDataVector[i] = ModelDataPtr(md);
	}

	// generate the skeleton of the model
	// specify the root bone
	if(m_BoneMapping.size() > 0 && m_modelType == RIGGED_MODEL)
	{
		Bone* skeleton_root = new Bone();
		skeleton_root->m_ID = 9999;
		skeleton_root->m_name = "Project Nebula Skeleton ROOT";

		mat4 identity;
		generateSkeleton(m_scene->mRootNode, skeleton_root, identity);
		m_skeleton = new Skeleton(skeleton_root, m_GlobalInverseTransform);

		// print out the skeleton
		//m_skeleton->dumpSkeleton(skeleton_root, 0);
	}

	// install the shader
	installShader();
	
	// prepare the vertex buffers (position, texcoord, normal, tangents...)
	prepareVertexBuffers();

	// print out the summary
	QString summary = "Loaded " + fileName + ". Model has " 
		            + QString::number(m_scene->mNumMeshes) + " meshes, " 
					+ QString::number(numVertices) + " vertices, " 
					+ QString::number(numIndices) + " indices.";

	if(m_NumBones)
		summary += " Contains " + QString::number(m_NumBones) + " bones.";
	if (m_scene->HasAnimations()) 
		summary += " Contains " + QString::number(m_scene->mAnimations[0]->mDuration) + " seconds animation.";

	qDebug() << summary;
	
	clear();
	return modelDataVector;
}

MeshData ModelLoader::loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh, const QString& fileName)
{
	MeshData data = MeshData();

	if(mesh->mName.length > 0)
		data.name = QString(mesh->mName.C_Str());
	else
		data.name = fileName + "/mesh_" + QString::number(index);

	data.numIndices = mesh->mNumFaces * 3;
	data.baseVertex = numVertices;
	data.baseIndex  = numIndices;

	prepareVertexContainers(index, mesh);

	return data;
}

void ModelLoader::prepareVertexContainers(unsigned int index, const aiMesh* mesh)
{
	const aiVector3D zero3D(0.0f, 0.0f, 0.0f);
	const aiColor4D  zeroColor(1.0f, 1.0f, 1.0f, 1.0f);

	// Populate the vertex attribute vectors
	for(unsigned int i = 0; i < mesh->mNumVertices; ++i)
	{
		const aiVector3D * pPos      = &(mesh->mVertices[i]);
		const aiColor4D  * pColor    = mesh->HasVertexColors(0)         ? &(mesh->mColors[0][i])        : &zeroColor;
		const aiVector3D * pTexCoord = mesh->HasTextureCoords(0)        ? &(mesh->mTextureCoords[0][i]) : &zero3D;
		const aiVector3D * pNormal   = mesh->HasNormals()               ? &(mesh->mNormals[i])          : &zero3D;
		const aiVector3D * pTangent  = mesh->HasTangentsAndBitangents() ? &(mesh->mTangents[i])         : &zero3D;
		
		m_positions.push_back(QVector3D(pPos->x, pPos->y, pPos->z));
		m_colors.push_back(QVector4D(pColor->r, pColor->g, pColor->b, pColor->a));
		m_texCoords.push_back(QVector2D(pTexCoord->x, pTexCoord->y));
		m_normals.push_back(QVector3D(pNormal->x, pNormal->y, pNormal->z));
		m_tangents.push_back(QVector3D(pTangent->x, pTangent->y, pTangent->z));
	}

	if(mesh->HasBones() && m_modelType == RIGGED_MODEL) loadBones(index, mesh);
	

	// Populate the index buffer
	for(unsigned int i = 0; i < mesh->mNumFaces; ++i)
	{
		const aiFace& face = mesh->mFaces[i];

		if(face.mNumIndices != 3)
		{
			// Unsupported modes : GL_POINTS / GL_LINES / GL_POLYGON
			qWarning(qPrintable(QObject::tr("Warning : unsupported number of indices per face (%1)").arg(face.mNumIndices)));
			break;
		}

		m_indices.push_back(face.mIndices[0]);
		m_indices.push_back(face.mIndices[1]);
		m_indices.push_back(face.mIndices[2]);

	}
}

void ModelLoader::installShader()
{
	QString shaderName, shaderPrefix, shaderFeatures;
	ShadingTechnique::ShaderType shaderType;

	// process shader prefix (skinning or static)
	if (m_modelType == RIGGED_MODEL)
	{
		shaderType = ShadingTechnique::RIGGED;
		shaderPrefix = "skinning";
	}
	else if (m_modelType == STATIC_MODEL)
	{
		shaderType = ShadingTechnique::STATIC;
		shaderPrefix = "static";
	}

	// process shader features (diffuse, bump, specular ...)
	if (m_modelFeatures.hasColorMap) shaderFeatures += "_diffuse";
	if (m_modelFeatures.hasNormalMap) shaderFeatures += "_bump";

	// combine the shader name
	shaderName = shaderPrefix + shaderFeatures;
	m_effect = new ShadingTechnique(shaderName, shaderType);

	if (!m_effect->Init()) 
	{
		qDebug() << shaderName << "may not be initialized successfully.";
	}
	m_shaderProgramID = m_effect->getShader()->programId();
}

void ModelLoader::prepareVertexBuffers()
{
	// Create the VAO
	glGenVertexArrays(1, &m_VAO);   
	glBindVertexArray(m_VAO);
	GLenum usage;
	if(m_modelType == RIGGED_MODEL) 
	{
		m_Buffers.resize(NUM_VBs);
		usage = GL_STREAM_DRAW;
	}
	else
	{
		m_Buffers.resize(NUM_VBs - 1);
		usage = GL_STATIC_DRAW;
	}

	// Create the buffers for the vertices attributes
	glGenBuffers(m_Buffers.size(), m_Buffers.data());

	// Generate and populate the buffers with vertex attributes and the indices
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[POS_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_positions[0]) * m_positions.size(), m_positions.data(), usage);
	GLuint POSITION_LOCATION = glGetAttribLocation(m_shaderProgramID, "Position");
	glEnableVertexAttribArray(POSITION_LOCATION);
	glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);    

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TEXCOORD_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_texCoords[0]) * m_texCoords.size(), m_texCoords.data(), usage);
	GLuint TEX_COORD_LOCATION = glGetAttribLocation(m_shaderProgramID, "TexCoord");
	glEnableVertexAttribArray(TEX_COORD_LOCATION);
	glVertexAttribPointer(TEX_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[NORMAL_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_normals[0]) * m_normals.size(), m_normals.data(), usage);
	GLuint NORMAL_LOCATION = glGetAttribLocation(m_shaderProgramID, "Normal");
	glEnableVertexAttribArray(NORMAL_LOCATION);
	glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TANGENT_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_tangents[0]) * m_tangents.size(), m_tangents.data(), usage);
	GLuint TANGENT_LOCATION = glGetAttribLocation(m_shaderProgramID, "Tangent");
	glEnableVertexAttribArray(TANGENT_LOCATION);
	glVertexAttribPointer(TANGENT_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Buffers[INDEX_BUFFER]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), usage);

	if (m_modelType == RIGGED_MODEL)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[BONE_VB]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(m_Bones[0]) * m_Bones.size(), m_Bones.data(), usage);

		GLuint BONE_ID_LOCATION = glGetAttribLocation(m_shaderProgramID, "BoneIDs");
		glEnableVertexAttribArray(BONE_ID_LOCATION);
		glVertexAttribIPointer(BONE_ID_LOCATION, 4, GL_INT, sizeof(VertexBoneData), (const GLvoid*)0);

		GLuint BONE_WEIGHT_LOCATION = glGetAttribLocation(m_shaderProgramID, "Weights");
		glEnableVertexAttribArray(BONE_WEIGHT_LOCATION);    
		glVertexAttribPointer(BONE_WEIGHT_LOCATION, 4, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*)16);
	}

	// Make sure the VAO is not changed from the outside
	glBindVertexArray(0);
	m_effect->setVAO(m_VAO);
}

void ModelLoader::loadBones( uint MeshIndex, const aiMesh* paiMesh )
{
	for (uint i = 0; i < paiMesh->mNumBones; ++i)
	{
		uint boneIndex = 0;        
		QString boneName(paiMesh->mBones[i]->mName.data);
		if (m_BoneMapping.find(boneName) == m_BoneMapping.end()) 
		{
			// Allocate an index for a new bone
			boneIndex = m_NumBones;
			m_NumBones++;
			Bone bi;
			m_BoneInfo.push_back(bi);
			m_BoneInfo[boneIndex].m_ID = boneIndex;
			m_BoneInfo[boneIndex].m_name = boneName;
			m_BoneInfo[boneIndex].m_offsetMatrix = Math::convToQMat4(paiMesh->mBones[i]->mOffsetMatrix);
			m_BoneMapping[boneName] = boneIndex;
		}
		else 
		{
			boneIndex = m_BoneMapping[boneName];
		}                      

		uint offset = 0;
		for (uint k = 0; k < MeshIndex; ++k)
		{
			offset += m_scene->mMeshes[k]->mNumVertices;
		}

		for (uint j = 0 ; j < paiMesh->mBones[i]->mNumWeights ; ++j) 
		{
			uint VertexID = offset + paiMesh->mBones[i]->mWeights[j].mVertexId;
			float Weight  = paiMesh->mBones[i]->mWeights[j].mWeight;                   
			m_Bones[VertexID].AddBoneData(boneIndex, Weight);
		}
	}
}

void ModelLoader::generateSkeleton( aiNode* pAiRootNode, Bone* pRootSkeleton, mat4& parentTransform )
{
	// generate a skeleton from the existing bone map and BoneInfo vector
	Bone* pBone = NULL;

	QString nodeName(pAiRootNode->mName.data);

	mat4 nodeTransformation(Math::convToQMat4(pAiRootNode->mTransformation));
	mat4 globalTransformation = parentTransform * nodeTransformation;

	// aiNode is not aiBone, aiBones are part of all the aiNodes
	if (m_BoneMapping.find(nodeName) != m_BoneMapping.end())
	{
		uint BoneIndex = m_BoneMapping[nodeName];
		m_BoneInfo[BoneIndex].m_boneSpaceTransform = Math::convToQMat4(pAiRootNode->mTransformation);

		Bone bi = m_BoneInfo[BoneIndex];
		pBone = new Bone(pRootSkeleton);
		pBone->m_ID = BoneIndex;
		pBone->m_name = bi.m_name;

		pBone->m_offsetMatrix = bi.m_offsetMatrix;
		pBone->m_boneSpaceTransform = Math::convToQMat4(pAiRootNode->mTransformation);
		pBone->m_modelSpaceTransform = globalTransformation;
		pBone->m_finalTransform = m_GlobalInverseTransform * pBone->m_modelSpaceTransform * pBone->m_offsetMatrix;
	}

	for (uint i = 0 ; i < pAiRootNode->mNumChildren ; ++i) 
	{
		if(pBone) generateSkeleton(pAiRootNode->mChildren[i], pBone, globalTransformation);
		else generateSkeleton(pAiRootNode->mChildren[i], pRootSkeleton, globalTransformation);
	}
}

MaterialData ModelLoader::loadMaterial(unsigned int index, const aiMaterial* material)
{
	Q_ASSERT(material != nullptr);

	MaterialData data = MaterialData();
	data.name = "material_" + QString::number(index);

	aiColor3D ambientColor(0.1f, 0.1f, 0.1f);
	aiColor3D diffuseColor(0.8f, 0.8f, 0.8f);
	aiColor3D specularColor(0.0f, 0.0f, 0.0f);
	aiColor3D emissiveColor(0.0f, 0.0f, 0.0f);

	int blendMode;
	int twoSided = 1;

	float opacity = 1.0f;
	float shininess = 0.0f;
	float shininessStrength = 1.0f;

	if(material->Get(AI_MATKEY_COLOR_AMBIENT, ambientColor) == AI_SUCCESS)
	{
		data.ambientColor.setX(ambientColor.r);
		data.ambientColor.setY(ambientColor.g);
		data.ambientColor.setZ(ambientColor.b);
	}

	if(material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor) == AI_SUCCESS)
	{
		data.diffuseColor.setX(diffuseColor.r);
		data.diffuseColor.setY(diffuseColor.g);
		data.diffuseColor.setZ(diffuseColor.b);
	}

	if(material->Get(AI_MATKEY_COLOR_SPECULAR, specularColor) == AI_SUCCESS)
	{
		data.specularColor.setX(specularColor.r);
		data.specularColor.setY(specularColor.g);
		data.specularColor.setZ(specularColor.b);
	}

	if(material->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor) == AI_SUCCESS)
	{
		data.emissiveColor.setX(emissiveColor.r);
		data.emissiveColor.setY(emissiveColor.g);
		data.emissiveColor.setZ(emissiveColor.b);
	}

	if(material->Get(AI_MATKEY_TWOSIDED, twoSided) == AI_SUCCESS)
	{
		data.twoSided = twoSided;
	}

	if(material->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
	{
		data.ambientColor.setW(opacity);
		data.diffuseColor.setW(opacity);
		data.specularColor.setW(opacity);
		data.emissiveColor.setW(opacity);

		if(opacity < 1.0f)
		{
			data.alphaBlending = true;

			// Activate backface culling allows to avoid
			// cull artifacts when alpha blending is activated
			data.twoSided = 1;

			if(material->Get(AI_MATKEY_BLEND_FUNC, blendMode) == AI_SUCCESS)
			{
				if(blendMode == aiBlendMode_Additive)
					data.blendMode = aiBlendMode_Additive;
				else
					data.blendMode = aiBlendMode_Default;
			}
		}
		else
		{
			data.alphaBlending = false;
			data.blendMode = -1;
		}
	}

	if(material->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS)
	{
		data.shininess = shininess;
	}

	if(material->Get(AI_MATKEY_SHININESS_STRENGTH, shininessStrength) == AI_SUCCESS)
	{
		data.shininessStrength = shininessStrength;
	}

	return data;
}

TextureData ModelLoader::loadTexture(const QString& fileName, const aiMaterial* material)
{
	Q_ASSERT(material != nullptr);

	// Extract the directory part from the file name
	int slashIndex = fileName.lastIndexOf("/");
	QString dir;


	if (slashIndex == std::string::npos) dir = ".";
	else if (slashIndex == 0) dir = "/";
	else dir = fileName.left(slashIndex);

	TextureData data = TextureData();
	aiString path;

	data.hasTexture = false;

//    if(material->GetTextureCount(aiTextureType_DIFFUSE)      > 0) qDebug() << "aiTextureType_DIFFUSE";
//    if(material->GetTextureCount(aiTextureType_SPECULAR)     > 0) qDebug() << "aiTextureType_SPECULAR";
//    if(material->GetTextureCount(aiTextureType_AMBIENT)      > 0) qDebug() << "aiTextureType_AMBIENT";
//    if(material->GetTextureCount(aiTextureType_EMISSIVE)     > 0) qDebug() << "aiTextureType_EMISSIVE";
//    if(material->GetTextureCount(aiTextureType_HEIGHT)       > 0) qDebug() << "aiTextureType_HEIGHT";
//    if(material->GetTextureCount(aiTextureType_NORMALS)      > 0) qDebug() << "aiTextureType_NORMALS";
//    if(material->GetTextureCount(aiTextureType_SHININESS)    > 0) qDebug() << "aiTextureType_SHININESS";
//    if(material->GetTextureCount(aiTextureType_OPACITY)      > 0) qDebug() << "aiTextureType_OPACITY";
//    if(material->GetTextureCount(aiTextureType_DISPLACEMENT) > 0) qDebug() << "aiTextureType_DISPLACEMENT";
//    if(material->GetTextureCount(aiTextureType_LIGHTMAP)     > 0) qDebug() << "aiTextureType_LIGHTMAP";
//    if(material->GetTextureCount(aiTextureType_REFLECTION)   > 0) qDebug() << "aiTextureType_REFLECTION";
//    if(material->GetTextureCount(aiTextureType_UNKNOWN)      > 0) qDebug() << "aiTextureType_UNKNOWN";

	if(material->GetTextureCount(aiTextureType_DIFFUSE) > 0)
	{
		if(material->GetTexture(aiTextureType_DIFFUSE, 0, &path) == AI_SUCCESS)
		{
			QString texturePath = dir + "/" + path.data;
			data.colorMap = texturePath;
			data.hasTexture = true;
			m_modelFeatures.hasColorMap = true;
		}
	}
	if(material->GetTextureCount(aiTextureType_NORMALS) > 0)
	{
		if(material->GetTexture(aiTextureType_NORMALS, 0, &path) == AI_SUCCESS)
		{
			QString texturePath = dir + "/" + path.data;
			data.normalMap = texturePath;
			m_modelFeatures.hasNormalMap = true;
		}
	}
	if(material->GetTextureCount(aiTextureType_OPACITY) > 0)
	{
		if(material->GetTexture(aiTextureType_OPACITY, 0, &path) == AI_SUCCESS)
		{
			QString texturePath = dir + "/" + path.data;
			data.colorMap = texturePath;
			data.hasTexture = true;
		}
	}

	return data;
}