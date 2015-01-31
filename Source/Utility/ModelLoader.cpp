#include "ModelLoader.h"
#include <QDebug>
#include <Scene/Scene.h>
#include <Utility/Math.h>
#include <Animation/IK/FABRIKSolver.h>

ModelLoader::ModelLoader(Scene* scene)
{
	m_scene = scene;
	m_effect = ShadingTechniquePtr();
	Q_ASSERT(initializeOpenGLFunctions());
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
}

ModelLoader::~ModelLoader()
{
	clear();
}

QVector<ModelDataPtr> ModelLoader::loadModel( const QString& fileName, GLuint shaderProgramID, const QString& loadingFlags )
{
	m_fileName = fileName;
	uint flags;

	if (loadingFlags == "Simple")
		flags = aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs;
	else if (loadingFlags == "Fast")
		flags = aiProcessPreset_TargetRealtime_Fast | aiProcess_FlipUVs;
	else if (loadingFlags == "Quality")
		flags = aiProcessPreset_TargetRealtime_Quality | aiProcess_FlipUVs;
	else if (loadingFlags == "Max Quality")
		flags = aiProcessPreset_TargetRealtime_MaxQuality | aiProcess_FlipUVs;

	qDebug() << "Processing model file...";
	m_aiScene = m_importer.ReadFile(fileName.toStdString(), flags);
	
	if(!m_aiScene)
	{
		qWarning() << m_importer.GetErrorString();
		QVector<ModelDataPtr> empty;
		return empty;
	}
	else if(m_aiScene->HasTextures())
	{
		qWarning() << "Support for meshes with embedded textures is not implemented";
		QVector<ModelDataPtr> empty;
		return empty;
	}
	
   	m_GlobalInverseTransform = Math::convToQMat4(m_aiScene->mRootNode->mTransformation.Inverse());

	m_modelType = m_aiScene->HasAnimations() ? RIGGED_MODEL : STATIC_MODEL;

	uint numVertices = 0, numIndices = 0, numFaces = 0;

	for(uint i = 0; i < m_aiScene->mNumMeshes; ++i)
	{
		numVertices += m_aiScene->mMeshes[i]->mNumVertices;
		numIndices  += m_aiScene->mMeshes[i]->mNumFaces * 3;
		numFaces    += m_aiScene->mMeshes[i]->mNumFaces;
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
	modelDataVector.resize(m_aiScene->mNumMeshes);

	for(uint i = 0; i < m_aiScene->mNumMeshes; ++i)
	{
		ModelDataPtr md(new ModelData());
		
		md->meshData     = loadMesh(i, numVertices, numIndices, m_aiScene->mMeshes[i]);
		md->materialData = loadMaterial(i, m_aiScene->mMaterials[m_aiScene->mMeshes[i]->mMaterialIndex]);
		md->hasAnimation = m_aiScene->HasAnimations();

		// calculate the animation duration in seconds
		if(m_aiScene->HasAnimations()) md->animationDuration = (float) m_aiScene->mAnimations[0]->mDuration;
		else md->animationDuration = 0.0f;

		numVertices += m_aiScene->mMeshes[i]->mNumVertices;
		numIndices  += m_aiScene->mMeshes[i]->mNumFaces * 3;

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
		generateSkeleton(m_aiScene->mRootNode, skeleton_root, identity);
		m_skeleton = new Skeleton(skeleton_root, m_GlobalInverseTransform);

		// print out the skeleton
		//m_skeleton->dumpSkeleton(skeleton_root, 0);
	}

	// install the shader
	m_shaderProgramID = shaderProgramID ? shaderProgramID : installShader();

	// prepare the vertex buffers (position, texcoord, normal, tangents...)
	prepareVertexBuffers();

	// print out the summary
	QString summary = "Loaded " + fileName + ". Model has " 
		            + QString::number(m_aiScene->mNumMeshes) + " meshes, " 
					+ QString::number(numVertices) + " vertices, " 
					+ QString::number(numFaces) + " faces.";

	if(m_NumBones)
		summary += " Contains " + QString::number(m_NumBones) + " bones.";
	if (m_aiScene->HasAnimations()) 
		summary += " Contains " + QString::number(m_aiScene->mAnimations[0]->mDuration) + " seconds animation.";

	qDebug() << summary;
	
	// clean up
	clear();
	return modelDataVector;
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
		
		m_positions << vec3(pPos->x, pPos->y, pPos->z);
		m_colors    << vec4(pColor->r, pColor->g, pColor->b, pColor->a);
		m_texCoords << vec2(pTexCoord->x, pTexCoord->y);
		m_normals   << vec3(pNormal->x, pNormal->y, pNormal->z);
		m_tangents  << vec3(pTangent->x, pTangent->y, pTangent->z);
	}

	if(mesh->HasBones() && m_modelType == RIGGED_MODEL) loadBones(index, mesh);
	

	// Populate the index buffer
	for(unsigned int i = 0; i < mesh->mNumFaces; ++i)
	{
		const aiFace& face = mesh->mFaces[i];

		if(face.mNumIndices != 3)
		{
			// Unsupported modes : GL_POINTS / GL_LINES / GL_POLYGON
			qWarning(qPrintable(QObject::tr("Unsupported number of indices per face (%1)").arg(face.mNumIndices)));
			break;
		}

		m_indices << (face.mIndices[0]);
		m_indices << (face.mIndices[1]);
		m_indices << (face.mIndices[2]);
	}
}

GLuint ModelLoader::installShader()
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
	m_effect = ShadingTechniquePtr(new ShadingTechnique(m_scene, shaderName, shaderType));

	return m_effect->getShaderProgram()->programId();
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
	GLuint POSITION_LOCATION = glGetAttribLocation(m_shaderProgramID, "Position");
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[POS_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_positions[0]) * m_positions.size(), m_positions.data(), usage);
	glEnableVertexAttribArray(POSITION_LOCATION);
	glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);    


	GLuint COLOR_LOCATION = glGetAttribLocation(m_shaderProgramID, "Color");
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[COLOR_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_colors[0]) * m_colors.size(), m_colors.data(), usage);
	glEnableVertexAttribArray(COLOR_LOCATION);
	glVertexAttribPointer(COLOR_LOCATION, 4, GL_FLOAT, GL_FALSE, 0, 0);


	GLuint TEX_COORD_LOCATION = glGetAttribLocation(m_shaderProgramID, "TexCoord");
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TEXCOORD_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_texCoords[0]) * m_texCoords.size(), m_texCoords.data(), usage);
	glEnableVertexAttribArray(TEX_COORD_LOCATION);
	glVertexAttribPointer(TEX_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, 0);

	GLuint NORMAL_LOCATION = glGetAttribLocation(m_shaderProgramID, "Normal");
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[NORMAL_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_normals[0]) * m_normals.size(), m_normals.data(), usage);
	glEnableVertexAttribArray(NORMAL_LOCATION);
	glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

	GLuint TANGENT_LOCATION = glGetAttribLocation(m_shaderProgramID, "Tangent");
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TANGENT_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_tangents[0]) * m_tangents.size(), m_tangents.data(), usage);
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
	if(m_effect) m_effect->setVAO(m_VAO);
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
			offset += m_aiScene->mMeshes[k]->mNumVertices;
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

MeshData ModelLoader::loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh)
{
	MeshData data;

	data.name = mesh->mName.length > 0
				? m_fileName + "/mesh_" + mesh->mName.C_Str()
				: m_fileName + "/mesh_" + QString::number(index);

	data.numIndices = mesh->mNumFaces * 3;
	data.baseVertex = numVertices;
	data.baseIndex  = numIndices;

	prepareVertexContainers(index, mesh);

	return data;
}

MaterialData ModelLoader::loadMaterial(unsigned int index, const aiMaterial* material)
{
	Q_ASSERT(material != nullptr);

	MaterialData data;
	data.name = m_fileName + "/material_" + QString::number(index);

	aiColor3D ambientColor(0.1f, 0.1f, 0.1f);
	aiColor3D diffuseColor(0.8f, 0.8f, 0.8f);
	aiColor3D specularColor(0.0f, 0.0f, 0.0f);
	aiColor3D emissiveColor(0.0f, 0.0f, 0.0f);
	data.ambientColor.setRgbF(ambientColor.r, ambientColor.g, ambientColor.b);
	data.diffuseColor.setRgbF(diffuseColor.r, diffuseColor.g, diffuseColor.b);
	data.specularColor.setRgbF(specularColor.r, specularColor.g, specularColor.b);
	data.emissiveColor.setRgbF(emissiveColor.r, emissiveColor.g, emissiveColor.b);

	int blendMode;
	data.blendMode = -1;

	int twoSided = 1;
	data.twoSided = 1;

	float opacity = 1.0f;
	
	float shininess = 10.0f;
	data.shininess = 10.0f;
	float shininessStrength = 0.0f;
	data.shininessStrength = 0.0f;

	if(material->Get(AI_MATKEY_COLOR_AMBIENT, ambientColor) == AI_SUCCESS)
	{
		data.ambientColor.setRgbF(ambientColor.r, ambientColor.g, ambientColor.b);
	}

	if(material->Get(AI_MATKEY_COLOR_DIFFUSE, diffuseColor) == AI_SUCCESS)
	{
		data.diffuseColor.setRgbF(diffuseColor.r, diffuseColor.g, diffuseColor.b);
	}

	if(material->Get(AI_MATKEY_COLOR_SPECULAR, specularColor) == AI_SUCCESS)
	{
		data.specularColor.setRgbF(specularColor.r, specularColor.g, specularColor.b);
	}

	if(material->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveColor) == AI_SUCCESS)
	{
		data.emissiveColor.setRgbF(emissiveColor.r, emissiveColor.g, emissiveColor.b);
	}

	if(material->Get(AI_MATKEY_TWOSIDED, twoSided) == AI_SUCCESS)
	{
		data.twoSided = twoSided;
	}

	if(material->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
	{
		data.ambientColor.setAlphaF(opacity);
		data.diffuseColor.setAlphaF(opacity);
		data.specularColor.setAlphaF(opacity);
		data.emissiveColor.setAlphaF(opacity);

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

	// process the textures
	data.textureData = loadTexture(material);

	return data;
}

TextureData ModelLoader::loadTexture(const aiMaterial* material)
{
	Q_ASSERT(material);

	// Extract the directory part from the file name
	QString absPath = QFileInfo(m_fileName).absolutePath();
	TextureData data = TextureData();
	aiString path;
	
	// process all texture types
	QVector<aiTextureType> textureTypes;
	if(material->GetTextureCount(aiTextureType_DIFFUSE)     ) textureTypes << aiTextureType_DIFFUSE;
	if(material->GetTextureCount(aiTextureType_SPECULAR)    ) textureTypes << aiTextureType_SPECULAR;
	if(material->GetTextureCount(aiTextureType_AMBIENT)     ) textureTypes << aiTextureType_AMBIENT;
	if(material->GetTextureCount(aiTextureType_EMISSIVE)    ) textureTypes << aiTextureType_EMISSIVE;
	if(material->GetTextureCount(aiTextureType_HEIGHT)      ) textureTypes << aiTextureType_HEIGHT;
	if(material->GetTextureCount(aiTextureType_NORMALS)     ) textureTypes << aiTextureType_NORMALS;
	if(material->GetTextureCount(aiTextureType_SHININESS)   ) textureTypes << aiTextureType_SHININESS;
	if(material->GetTextureCount(aiTextureType_OPACITY)     ) textureTypes << aiTextureType_OPACITY;
	if(material->GetTextureCount(aiTextureType_DISPLACEMENT)) textureTypes << aiTextureType_DISPLACEMENT;
	if(material->GetTextureCount(aiTextureType_LIGHTMAP)    ) textureTypes << aiTextureType_LIGHTMAP;
	if(material->GetTextureCount(aiTextureType_REFLECTION)  ) textureTypes << aiTextureType_REFLECTION;
	//if(material->GetTextureCount(aiTextureType_UNKNOWN)     ) textureFeatures << aiTextureType_UNKNOWN;

	foreach(aiTextureType type, textureTypes)
	{
		if(material->GetTexture(type, 0, &path) == AI_SUCCESS)
		{
			QString texturePath = absPath + "/" + path.data;

			if (type == aiTextureType_DIFFUSE)
			{
				data.diffuseMap = texturePath;
				m_modelFeatures.hasColorMap = true;
			}
			else if (type == aiTextureType_NORMALS)
			{
				data.normalMap = texturePath;
				m_modelFeatures.hasNormalMap = true;
			}
			else if (type == aiTextureType_OPACITY)
			{
				data.opacityMap = texturePath;
			}
		}
	}

	return data;
}
