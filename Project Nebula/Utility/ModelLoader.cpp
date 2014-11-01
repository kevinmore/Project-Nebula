#include "ModelLoader.h"
#include <QtCore/QDebug>
#include <Utility/Math.h>



ModelLoader::ModelLoader()
	: m_vao(new QOpenGLVertexArrayObject())
{
	initializeOpenGLFunctions();
	clear();
}


QOpenGLVertexArrayObject* ModelLoader::getVAO()
{
	return m_vao;
}

void ModelLoader::clear()
{
	m_positions.clear();
	m_colors.clear();
	m_texCoords.clear();
	m_normals.clear();
	m_tangents.clear();
	m_indices.clear();

	if (m_Buffers[0] != 0) {
		glDeleteBuffers(ARRAY_SIZE_IN_ELEMENTS(m_Buffers), m_Buffers);
	}

	if (m_vao)
	{
		m_vao->destroy();
	}
}

ModelLoader::~ModelLoader()
{
	clear();
}

QVector<ModelDataPtr> ModelLoader::loadModel( const QString& fileName )
{
	m_scene = m_importer.ReadFile(fileName.toStdString(),	
		  aiProcess_Triangulate
		| aiProcess_GenSmoothNormals
		| aiProcess_FlipUVs
 		| aiProcess_CalcTangentSpace
 		| aiProcess_JoinIdenticalVertices
 		| aiProcess_SortByPType
		| aiProcess_LimitBoneWeights
		| aiProcess_FixInfacingNormals
// 		| aiProcessPreset_TargetRealtime_MaxQuality
		);

	if(!m_scene)
	{
		qDebug() << "Error loading mesh file: " << fileName << m_importer.GetErrorString();
	}
	else if(m_scene->HasTextures())
	{
		qFatal("Support for meshes with embedded textures is not implemented");
	}


  	m_GlobalInverseTransform = Math::convToQMat4(&m_scene->mRootNode->mTransformation);
//  	inverseQMat4(m_GlobalInverseTransform);
// 	m_GlobalInverseTransform = mat4(1, 0, 0, 0, 
// 									0, 0, -1, 0,
// 									0, 1, 0, 0,
// 									0, 0, 0, 1);
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
	m_Bones.resize(numVertices);

	numVertices = 0;
	numIndices  = 0;
	m_NumBones = 0;

	QVector<ModelDataPtr> modelDataVector;
	modelDataVector.resize(m_scene->mNumMeshes);

	for(uint i = 0; i < m_scene->mNumMeshes; ++i)
	{
		ModelData* md = new ModelData();
		
		md->meshData     = loadMesh(i, numVertices, numIndices, m_scene->mMeshes[i]);
		md->textureData  = loadTexture(fileName, m_scene->mMaterials[m_scene->mMeshes[i]->mMaterialIndex]);
		md->materialData = loadMaterial(i, m_scene->mMaterials[m_scene->mMeshes[i]->mMaterialIndex]);
		md->hasAnimation = m_scene->HasAnimations();

		numVertices += m_scene->mMeshes[i]->mNumVertices;
		numIndices  += m_scene->mMeshes[i]->mNumFaces * 3;
		
		modelDataVector[i] = ModelDataPtr(md);
	}

	prepareVertexBuffers();

	qDebug() << "Loaded" << fileName;
	qDebug() << "Model has" << m_scene->mNumMeshes << "meshes," << numVertices << "vertices," << numIndices << "indices and" << m_NumBones << "bones.";

	return modelDataVector;
}

MeshData ModelLoader::loadMesh(unsigned int index, unsigned int numVertices, unsigned int numIndices, const aiMesh* mesh)
{
	MeshData data = MeshData();

	if(mesh->mName.length > 0)
		data.name = QString(mesh->mName.C_Str());
	else
		data.name = "mesh_" + QString::number(index);

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

	if(mesh->HasBones()) loadBones(index, mesh);
	

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

void ModelLoader::prepareVertexBuffers()
{
	m_vao->create();
	m_vao->bind();
	glGenBuffers(ARRAY_SIZE_IN_ELEMENTS(m_Buffers), m_Buffers);

	// Generate and populate the buffers with vertex attributes and the indices
	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[POS_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_positions[0]) * m_positions.size(), m_positions.data(), GL_STREAM_DRAW);
	glEnableVertexAttribArray(POSITION_LOCATION);
	glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);    

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TEXCOORD_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_texCoords[0]) * m_texCoords.size(), m_texCoords.data(), GL_STREAM_DRAW);
	glEnableVertexAttribArray(TEX_COORD_LOCATION);
	glVertexAttribPointer(TEX_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[NORMAL_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_normals[0]) * m_normals.size(), m_normals.data(), GL_STREAM_DRAW);
	glEnableVertexAttribArray(NORMAL_LOCATION);
	glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[BONE_VB]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(m_Bones[0]) * m_Bones.size(), m_Bones.data(), GL_STREAM_DRAW);
	glEnableVertexAttribArray(BONE_ID_LOCATION);
	glVertexAttribIPointer(BONE_ID_LOCATION, 4, GL_INT, sizeof(VertexBoneData), (const GLvoid*)0);
	glEnableVertexAttribArray(BONE_WEIGHT_LOCATION);    
	glVertexAttribPointer(BONE_WEIGHT_LOCATION, 4, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*)16);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Buffers[INDEX_BUFFER]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STREAM_DRAW);

	m_vao->release();
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
			data.filename = texturePath;
			data.hasTexture = true;
		}
	}
	else if(material->GetTextureCount(aiTextureType_OPACITY) > 0)
	{
		if(material->GetTexture(aiTextureType_OPACITY, 0, &path) == AI_SUCCESS)
		{
			QString texturePath = dir + "/" + path.data;
			data.filename = texturePath;
			data.hasTexture = true;
		}
	}

	return data;
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
			BoneInfo bi;			
			m_BoneInfo.push_back(bi);
			m_BoneInfo[boneIndex].boneOffset = Math::convToQMat4(&paiMesh->mBones[i]->mOffsetMatrix);
			m_BoneMapping[boneName] = boneIndex;
			//qDebug() << "Loaded Bone:" << boneName;
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
