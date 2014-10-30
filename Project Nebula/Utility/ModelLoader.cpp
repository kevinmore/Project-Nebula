#include "ModelLoader.h"
#include <QtCore/QDebug>

// NOT SURE WHICH ONE TO USE!!!!!!!!!!!!
// utility function to convert aiMatrix4x4 to QMatrix4x4
QMatrix4x4 convToQMat4(const aiMatrix4x4 * m)
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
					  0,     0,     0,     1);

}

// QMatrix4x4 convToQMat4(const aiMatrix4x4 * m) {
// 	return QMatrix4x4(m->a1, m->b1, m->c1, m->d1,
// 					  m->a2, m->b2, m->c2, m->d2,
// 					  m->a3, m->b3, m->c3, m->d3,
// 					  m->a4, m->b4, m->c4, m->d4);
// }
// 
// QMatrix4x4 convToQMat4(aiMatrix3x3 * m) {
// 	return QMatrix4x4(m->a1, m->b1, m->c1, 0,
// 					  m->a2, m->b2, m->c2, 0,
// 					  m->a3, m->b3, m->c3, 0,
// 					  0, 0, 0, 1);
// }


void inverseQMat4(QMatrix4x4 &m)
{
	float det = m.determinant();
	if(det == 0.0f) 
	{
		// Matrix not invertible. Setting all elements to nan is not really
		// correct in a mathematical sense but it is easy to debug for the
		// programmer.
		/*const float nan = std::numeric_limits<float>::quiet_NaN();
		*this = Matrix4f(
			nan,nan,nan,nan,
			nan,nan,nan,nan,
			nan,nan,nan,nan,
			nan,nan,nan,nan);*/
		return;
	}
	qDebug() << "Input:" << endl << m;
	float invdet = 1.0f / det;

	QMatrix4x4 res;
	
	res(0, 0) = invdet  * (m(1, 1) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2) + m(1, 2) * (m(2, 3) * m(3, 1) - m(2, 1) * m(3, 3)) + m(1, 3) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1))));
	res(0, 1) = -invdet * (m(0, 1) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2) + m(0, 2) * (m(2, 3) * m(3, 1) - m(2, 1) * m(3, 3)) + m(0, 3) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1))));
	res(0, 2) = invdet  * (m(0, 1) * (m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2) + m(0, 2) * (m(1, 3) * m(3, 1) - m(1, 1) * m(3, 3)) + m(0, 3) * (m(1, 1) * m(3, 2) - m(1, 2) * m(3, 1))));
	res(0, 3) = -invdet * (m(0, 1) * (m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2) + m(0, 2) * (m(1, 3) * m(2, 1) - m(1, 1) * m(2, 3)) + m(0, 3) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))));
	res(1, 0) = -invdet * (m(1, 0) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2) + m(1, 2) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)) + m(1, 3) * (m(2, 0) * m(3, 2) - m(2, 2) * m(3, 0))));
	res(1, 1) = invdet  * (m(0, 0) * (m(2, 2) * m(3, 3) - m(2, 3) * m(3, 2) + m(0, 2) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)) + m(0, 3) * (m(2, 0) * m(3, 2) - m(2, 2) * m(3, 0))));
	res(1, 2) = -invdet * (m(0, 0) * (m(1, 2) * m(3, 3) - m(1, 3) * m(3, 2) + m(0, 2) * (m(1, 3) * m(3, 0) - m(1, 0) * m(3, 3)) + m(0, 3) * (m(1, 0) * m(3, 2) - m(1, 2) * m(3, 0))));
	res(1, 3) = invdet  * (m(0, 0) * (m(1, 2) * m(2, 3) - m(1, 3) * m(2, 2) + m(0, 2) * (m(1, 3) * m(2, 0) - m(1, 0) * m(2, 3)) + m(0, 3) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))));
	res(2, 0) = invdet  * (m(1, 0) * (m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1) + m(1, 1) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)) + m(1, 3) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0))));
	res(2, 1) = -invdet * (m(0, 0) * (m(2, 1) * m(3, 3) - m(2, 3) * m(3, 1) + m(0, 1) * (m(2, 3) * m(3, 0) - m(2, 0) * m(3, 3)) + m(0, 3) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0))));
	res(2, 2) = invdet  * (m(0, 0) * (m(1, 1) * m(3, 3) - m(1, 3) * m(3, 1) + m(0, 1) * (m(1, 3) * m(3, 0) - m(1, 0) * m(3, 3)) + m(0, 3) * (m(1, 0) * m(3, 1) - m(1, 1) * m(3, 0))));
	res(2, 3) = -invdet * (m(0, 0) * (m(1, 1) * m(2, 3) - m(1, 3) * m(2, 1) + m(0, 1) * (m(1, 3) * m(2, 0) - m(1, 0) * m(2, 3)) + m(0, 3) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))));
	res(3, 0) = -invdet * (m(1, 0) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1) + m(1, 1) * (m(2, 2) * m(3, 0) - m(2, 0) * m(3, 2)) + m(1, 2) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0))));
	res(3, 1) = invdet  * (m(0, 0) * (m(2, 1) * m(3, 2) - m(2, 2) * m(3, 1) + m(0, 1) * (m(2, 2) * m(3, 0) - m(2, 0) * m(3, 2)) + m(0, 2) * (m(2, 0) * m(3, 1) - m(2, 1) * m(3, 0))));
	res(3, 2) = -invdet * (m(0, 0) * (m(1, 1) * m(3, 2) - m(1, 2) * m(3, 1) + m(0, 1) * (m(1, 2) * m(3, 0) - m(1, 0) * m(3, 2)) + m(0, 2) * (m(1, 0) * m(3, 1) - m(1, 1) * m(3, 0))));
	res(3, 3) = invdet  * (m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1) + m(0, 1) * (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)))); 
	
	

	qDebug() << "Output:" << endl << res;
	m = QMatrix4x4(res);
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
{
	QOpenGLContext* context = QOpenGLContext::currentContext();

	Q_ASSERT(context);

	m_funcs = context->versionFunctions<QOpenGLFunctions_4_3_Core>();
	m_funcs->initializeOpenGLFunctions();
}


QOpenGLVertexArrayObjectPtr ModelLoader::getVAO()
{
	return m_vao;
}

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

	m_scene = m_importer.ReadFile(fileName.toStdString(),	
		  aiProcess_Triangulate
		| aiProcess_GenSmoothNormals
		| aiProcess_FlipUVs
 		| aiProcess_CalcTangentSpace
 		| aiProcess_JoinIdenticalVertices
 		| aiProcess_SortByPType
// 		| aiProcessPreset_TargetRealtime_MaxQuality
		);

	if(!m_scene)
	{
		qDebug() << "Error loading mesh file: " << fileName << m_importer.GetErrorString();
	}
	else if(m_scene->HasTextures())
	{
		qFatal("Support for meshes with embedded textures is not implemented");
		exit(1);
	}


// 	m_GlobalInverseTransform = convToQMat4(&m_scene->mRootNode->mTransformation);
// 	inverseQMat4(m_GlobalInverseTransform);
	m_GlobalInverseTransform = mat4(1, 0, 0, 0, 
									0, 0, -1, 0,
									0, 1, 0, 0,
									0, 0, 0, 1);
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

		numVertices += m_scene->mMeshes[i]->mNumVertices;
		numIndices  += m_scene->mMeshes[i]->mNumFaces * 3;
		
		modelDataVector[i] = ModelDataPtr(md);
	}

	prepareVertexBuffers();

	qDebug() << "Model has" << m_scene->mNumMeshes << "meshes," << numVertices << "vertices, and" << numIndices << "indices";

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

	}
}

void ModelLoader::prepareVertexBuffers()
{
	m_vao->create();
	m_vao->bind();
	m_funcs->glGenBuffers(ARRAY_SIZE_IN_ELEMENTS(m_Buffers), m_Buffers);

	// Generate and populate the buffers with vertex attributes and the indices
	m_funcs->glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[POS_VB]);
	m_funcs->glBufferData(GL_ARRAY_BUFFER, sizeof(m_positions[0]) * m_positions.size(), m_positions.data(), GL_STREAM_DRAW);
	m_funcs->glEnableVertexAttribArray(POSITION_LOCATION);
	m_funcs->glVertexAttribPointer(POSITION_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);    

	m_funcs->glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[TEXCOORD_VB]);
	m_funcs->glBufferData(GL_ARRAY_BUFFER, sizeof(m_texCoords[0]) * m_texCoords.size(), m_texCoords.data(), GL_STREAM_DRAW);
	m_funcs->glEnableVertexAttribArray(TEX_COORD_LOCATION);
	m_funcs->glVertexAttribPointer(TEX_COORD_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, 0);

	m_funcs->glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[NORMAL_VB]);
	m_funcs->glBufferData(GL_ARRAY_BUFFER, sizeof(m_normals[0]) * m_normals.size(), m_normals.data(), GL_STREAM_DRAW);
	m_funcs->glEnableVertexAttribArray(NORMAL_LOCATION);
	m_funcs->glVertexAttribPointer(NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, 0);

	m_funcs->glBindBuffer(GL_ARRAY_BUFFER, m_Buffers[BONE_VB]);
	m_funcs->glBufferData(GL_ARRAY_BUFFER, sizeof(m_Bones[0]) * m_Bones.size(), m_Bones.data(), GL_STREAM_DRAW);
	m_funcs->glEnableVertexAttribArray(BONE_ID_LOCATION);
	m_funcs->glVertexAttribIPointer(BONE_ID_LOCATION, 4, GL_INT, sizeof(VertexBoneData), (const GLvoid*)0);
	m_funcs->glEnableVertexAttribArray(BONE_WEIGHT_LOCATION);    
	m_funcs->glVertexAttribPointer(BONE_WEIGHT_LOCATION, 4, GL_FLOAT, GL_FALSE, sizeof(VertexBoneData), (const GLvoid*)16);

	m_funcs->glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Buffers[INDEX_BUFFER]);
	m_funcs->glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices[0]) * m_indices.size(), m_indices.data(), GL_STREAM_DRAW);

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

void ModelLoader::LoadBones( uint MeshIndex, const aiMesh* paiMesh )
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

void ModelLoader::CalcInterpolatedPosition(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
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
	//assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiVector3D& Start = pNodeAnim->mPositionKeys[PositionIndex].mValue;
	const aiVector3D& End = pNodeAnim->mPositionKeys[NextPositionIndex].mValue;
	aiVector3D Delta = End - Start;
	Out = Start + Factor * Delta;
}


void ModelLoader::CalcInterpolatedRotation(aiQuaternion& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
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
	//assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiQuaternion& StartRotationQ = pNodeAnim->mRotationKeys[RotationIndex].mValue;
	const aiQuaternion& EndRotationQ   = pNodeAnim->mRotationKeys[NextRotationIndex].mValue;    
	aiQuaternion::Interpolate(Out, StartRotationQ, EndRotationQ, Factor);
	Out = Out.Normalize();
}


void ModelLoader::CalcInterpolatedScaling(aiVector3D& Out, float AnimationTime, const aiNodeAnim* pNodeAnim)
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
	//assert(Factor >= 0.0f && Factor <= 1.0f);
	const aiVector3D& Start = pNodeAnim->mScalingKeys[ScalingIndex].mValue;
	const aiVector3D& End   = pNodeAnim->mScalingKeys[NextScalingIndex].mValue;
	aiVector3D Delta = End - Start;
	Out = Start + Factor * Delta;
}

uint ModelLoader::FindPosition(float AnimationTime, const aiNodeAnim* pNodeAnim)
{    
	for (uint i = 0 ; i < pNodeAnim->mNumPositionKeys - 1 ; i++) {
		if (AnimationTime < (float)pNodeAnim->mPositionKeys[i + 1].mTime) {
			return i;
		}
	}

	assert(0);

	return 0;
}


uint ModelLoader::FindRotation(float AnimationTime, const aiNodeAnim* pNodeAnim)
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


uint ModelLoader::FindScaling(float AnimationTime, const aiNodeAnim* pNodeAnim)
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

const aiNodeAnim* ModelLoader::FindNodeAnim(const aiAnimation* pAnimation, const QString NodeName)
{
	for (uint i = 0 ; i < pAnimation->mNumChannels ; i++) {
		const aiNodeAnim* pNodeAnim = pAnimation->mChannels[i];

		if (QString(pNodeAnim->mNodeName.data) == NodeName) {
			return pNodeAnim;
		}
	}

	return NULL;
}

void ModelLoader::ReadNodeHeirarchy( float AnimationTime, const aiNode* pNode, const mat4 &ParentTransform )
{
	QString NodeName(pNode->mName.data);

	const aiAnimation* pAnimation = m_scene->mAnimations[0];

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

void ModelLoader::BoneTransform( float TimeInSeconds, QVector<mat4>& Transforms )
{
	if(!m_scene->HasAnimations()) return;

	mat4 Identity;
	Identity.setToIdentity();
	float TicksPerSecond = (float)(m_scene->mAnimations[0]->mTicksPerSecond != 0 ? m_scene->mAnimations[0]->mTicksPerSecond : 25.0f);
	float TimeInTicks = TimeInSeconds * TicksPerSecond;
	float AnimationTime = fmod(TimeInTicks, (float)m_scene->mAnimations[0]->mDuration);
	ReadNodeHeirarchy(AnimationTime, m_scene->mRootNode, Identity);

	Transforms.resize(m_NumBones);

	for (uint i = 0 ; i < m_NumBones ; i++) {
		Transforms[i] = m_BoneInfo[i].finalTransformation;
	}
}
