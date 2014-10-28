#include "ModelLoader.h"
#include <QtCore/QDebug>
#include <assert.h>
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

void ModelLoader::VertexBoneData::AddBoneData(uint BoneID, float Weight)
{
	for (uint i = 0 ; i < ARRAY_SIZE_IN_ELEMENTS(IDs) ; i++) {
		if (Weights[i] == 0.0) {
			IDs[i]     = BoneID;
			Weights[i] = Weight;
			return;
		}        
	}

	// should never get here - more bones than we have space for
	assert(0);
}

ModelLoader::ModelLoader()
	: m_vertexPositionBuffer(QOpenGLBuffer::VertexBuffer),
	  m_vertexColorBuffer(QOpenGLBuffer::VertexBuffer),
	  m_vertexTexCoordBuffer(QOpenGLBuffer::VertexBuffer),
	  m_vertexNormalBuffer(QOpenGLBuffer::VertexBuffer),
	  m_vertexTangentBuffer(QOpenGLBuffer::VertexBuffer),
	  m_vertexBoneBuffer(QOpenGLBuffer::VertexBuffer),
	  m_indexBuffer(QOpenGLBuffer::IndexBuffer),
	  m_vao(QOpenGLVertexArrayObjectPtr(new QOpenGLVertexArrayObject()))
{}


ModelLoader::~ModelLoader()
{
	m_vertexPositionBuffer.destroy();
	m_vertexColorBuffer.destroy();
	m_vertexTexCoordBuffer.destroy();
	m_vertexNormalBuffer.destroy();
	m_vertexTangentBuffer.destroy();
	m_indexBuffer.destroy();

	m_vao->destroy();
}

QVector<ModelDataPtr> ModelLoader::loadModel( const QString& fileName, const QOpenGLShaderProgramPtr& shaderProgram )
{
	m_shaderProgram = shaderProgram;

	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(fileName.toStdString(),	
		  aiProcess_Triangulate
		| aiProcess_GenSmoothNormals
		| aiProcess_FlipUVs
 		| aiProcess_CalcTangentSpace
 		| aiProcess_JoinIdenticalVertices
 		| aiProcess_SortByPType
// 		| aiProcessPreset_TargetRealtime_MaxQuality
		);

	if(!scene)
	{
		qDebug() << "Error loading mesh file: " << fileName << importer.GetErrorString();
	}
	else if(scene->HasTextures())
	{
		qFatal("Support for meshes with embedded textures is not implemented");
		exit(1);
	}


	qDebug() << "Model has" << scene->mNumMeshes << "meshes";

	m_scene = scene;
	m_GlobalInverseTransform = convToQMat4(&scene->mRootNode->mTransformation);
	m_GlobalInverseTransform.inverted();

	unsigned int numVertices = 0;
	unsigned int numIndices  = 0;

	for(uint i = 0; i < scene->mNumMeshes; ++i)
	{
		numVertices += scene->mMeshes[i]->mNumVertices;
		numIndices  += scene->mMeshes[i]->mNumFaces * 3;
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
	modelDataVector.resize(scene->mNumMeshes);

	for(uint i = 0; i < scene->mNumMeshes; ++i)
	{
		ModelData* md = new ModelData();
		
		md->meshData     = loadMesh(i, numVertices, numIndices, scene->mMeshes[i]);
		md->textureData  = loadTexture(fileName, scene->mMaterials[scene->mMeshes[i]->mMaterialIndex]);
		md->materialData = loadMaterial(i, scene->mMaterials[scene->mMeshes[i]->mMaterialIndex]);

		numVertices += scene->mMeshes[i]->mNumVertices;
		numIndices  += scene->mMeshes[i]->mNumFaces * 3;
		
		modelDataVector[i] = ModelDataPtr(md);
	}

	prepareVertexBuffers();

	qDebug() << "Model has" << numVertices << "vertices";
	qDebug() << "Model has" << numIndices << "indices";

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

	LoadBones(index, mesh);
	

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

		// Voir : http://gamedev.stackexchange.com/q/45683
	}
}

void ModelLoader::prepareVertexBuffers()
{
	m_vao->create();
	m_vao->bind();


	// Generate and populate the buffers with vertex attributes and the indices
	m_vertexPositionBuffer.create();
	m_vertexPositionBuffer.bind();
	m_vertexPositionBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_vertexPositionBuffer.allocate(m_positions.data(), m_positions.size() * sizeof(QVector3D));

	m_vertexColorBuffer.create();
	m_vertexColorBuffer.bind();
	m_vertexColorBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_vertexColorBuffer.allocate(m_colors.data(), m_colors.size() * sizeof(QVector4D));

	m_vertexTexCoordBuffer.create();
	m_vertexTexCoordBuffer.bind();
	m_vertexTexCoordBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_vertexTexCoordBuffer.allocate(m_texCoords.data(), m_texCoords.size() * sizeof(QVector2D));

	m_vertexNormalBuffer.create();
	m_vertexNormalBuffer.bind();
	m_vertexNormalBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_vertexNormalBuffer.allocate(m_normals.data(), m_normals.size() * sizeof(QVector3D));

	m_vertexTangentBuffer.create();
	m_vertexTangentBuffer.bind();
	m_vertexTangentBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_vertexTangentBuffer.allocate(m_tangents.data(), m_tangents.size() * sizeof(QVector3D));

	m_vertexBoneBuffer.create();
	m_vertexBoneBuffer.bind();
	m_vertexBoneBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_vertexBoneBuffer.allocate(m_Bones.data(), m_Bones.size() * sizeof(VertexBoneData));

	m_indexBuffer.create();
	m_indexBuffer.bind();
	m_indexBuffer.setUsagePattern(QOpenGLBuffer::StreamDraw);
	m_indexBuffer.allocate(m_indices.data(), m_indices.size() * sizeof(unsigned int));


	m_shaderProgram->bind();

	m_vertexPositionBuffer.bind();
	m_shaderProgram->enableAttributeArray("position");
	m_shaderProgram->setAttributeBuffer("position", GL_FLOAT, 0, 3);
	m_vertexPositionBuffer.release();

	m_vertexColorBuffer.bind();
	m_shaderProgram->enableAttributeArray("color");
	m_shaderProgram->setAttributeBuffer("color", GL_FLOAT, 0, 4);
	m_vertexColorBuffer.release();

	m_vertexTexCoordBuffer.bind();
	m_shaderProgram->enableAttributeArray("texCoord");
	m_shaderProgram->setAttributeBuffer("texCoord", GL_FLOAT, 0, 2);
	m_vertexTexCoordBuffer.release();

	m_vertexNormalBuffer.bind();
	m_shaderProgram->enableAttributeArray("normal");
	m_shaderProgram->setAttributeBuffer("normal", GL_FLOAT, 0, 3);
	m_vertexNormalBuffer.release();

	m_vertexTangentBuffer.bind();
	m_shaderProgram->enableAttributeArray("tangent");
	m_shaderProgram->setAttributeBuffer("tangent", GL_FLOAT, 0, 3);
	m_vertexTangentBuffer.release();

	m_vertexBoneBuffer.bind();
	m_shaderProgram->enableAttributeArray("BoneIDs");
	m_shaderProgram->setAttributeBuffer("BoneIDs", GL_INT, 0, 4);
	m_shaderProgram->enableAttributeArray("Weights");
	m_shaderProgram->setAttributeBuffer("Weights", GL_FLOAT, 16, 4);
	m_vertexBoneBuffer.release();

	m_shaderProgram->release();
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

QOpenGLVertexArrayObjectPtr ModelLoader::getVAO()
{
	return m_vao;
}

void ModelLoader::LoadBones( uint MeshIndex, const aiMesh* paiMesh )
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
