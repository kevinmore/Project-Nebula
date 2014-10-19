#include "MeshImporter.h"



MeshImporter::MeshImporter(void)
{
	cleanUp();
	m_Root = new Bone();
}


MeshImporter::~MeshImporter(void)
{
}



// utility function to convert aiMatrix4x4 to QMatrix4x4
QMatrix4x4 conv(const aiMatrix4x4 * m) {
	return QMatrix4x4(m->a1, m->b1, m->c1, m->d1,
		m->a2, m->b2, m->c2, m->d2,
		m->a3, m->b3, m->c3, m->d3,
		m->a4, m->b4, m->c4, m->d4);
}

void MeshImporter::cleanUp()
{
	m_vertices.clear();
	m_normals.clear();
	m_indices.clear();
	m_materials.clear();
	m_meshes.clear();
	m_Root = 0;
}


bool MeshImporter::loadMeshFromFile( const QString &fileName )
{
	Assimp::Importer importer;

	const aiScene* scene = importer.ReadFile(fileName.toStdString(),
		aiProcess_GenSmoothNormals |
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType |
		aiProcess_FlipUVs
		);

	if (!scene)
	{
		qDebug() << "Error loading mesh file: " << fileName << importer.GetErrorString();
		return false;
	}

	// Materials must be loaded before meshes, and meshes must be loaded before bones.

	// load materials
	if (scene->HasMaterials())
	{
		for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
		{
			MaterialInfo* mater = processMaterial(scene->mMaterials[i], fileName);
			m_materials.push_back(mater);
		}
	}

	// load mesh
	if (scene->HasMeshes())
	{
		for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
		{
			m_meshes.push_back(processMesh(scene->mMeshes[i]));
		}
	}
	else
	{
		qDebug() << "Error: No meshes found" << fileName;
		return false;
	}

	// load bones
	if (scene->mRootNode != NULL)
	{
		processSkeleton(scene, scene->mRootNode, 0, *m_Root);
	}
	else
	{
		qDebug() << "Error: The model has no root node" << fileName;
		return false;
	}

	return true;
}

MaterialInfo* MeshImporter::processMaterial( aiMaterial *pMaterial, const QString &fileName )
{
	MaterialInfo* mater(new MaterialInfo);
	aiString mname;
	pMaterial->Get(AI_MATKEY_NAME, mname);
	if (mname.length > 0)
		mater->Name = mname.C_Str();


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
	
	if (pMaterial->GetTextureCount(aiTextureType_DIFFUSE) > 0) 
	{
		aiString Path;

		if (pMaterial->GetTexture(aiTextureType_DIFFUSE, 0, &Path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) 
		{
			QString FullPath = Dir + "/" + Path.data;
			mater->textureFile = FullPath;
		}
	}


	int shadingModel;
	pMaterial->Get(AI_MATKEY_SHADING_MODEL, shadingModel);

	if (shadingModel != aiShadingMode_Phong && shadingModel != aiShadingMode_Gouraud)
	{
		qDebug() << "This mesh's shading model is not implemented in this loader, setting to default material";
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

MeshData* MeshImporter::processMesh( aiMesh *mesh )
{
	MeshData* ret(new MeshData);

	ret->meshName = mesh->mName.length != 0 ? mesh->mName.C_Str() : "";
	ret->numVertices = mesh->mNumVertices;
	ret->vertices = new Vertex[ret->numVertices];

	// Get vertex attributes
	for (uint i = 0; i < mesh->mNumVertices; ++i)
	{
		// position
		ret->vertices[i].postition = vec3 (mesh->mVertices[i].x, 
											mesh->mVertices[i].y, 
											mesh->mVertices[i].z);

		// Color
		if (mesh->HasVertexColors(0))
		{
			ret->vertices[i].color = vec4(mesh->mColors[i]->r, 
											mesh->mColors[i]->g, 
											mesh->mColors[i]->b, 
											mesh->mColors[i]->a);
		}else{ret->vertices[i].color = vec4(0.5, 0, 0.5, 1);}

		// normals
		if (mesh->HasNormals())
		{
			ret->vertices[i].normal = vec3 ( mesh->mNormals[i].x,  
											mesh->mNormals[i].y,  
											mesh->mNormals[i].z);
		}

		if (mesh->HasTextureCoords(0))
		{
			ret->vertices[i].texCoord = vec2 (mesh->mTextureCoords[0][i].x, 
												mesh->mTextureCoords[0][i].y);
		}
	}


	// Get mesh indexes
	ret->numIndices = mesh->mNumFaces*3;
	ret->indices = new GLushort[ret->numIndices];
	for (uint i = 0; i < mesh->mNumFaces; ++i)
	{
		aiFace* face = &mesh->mFaces[i];
		if (face->mNumIndices != 3)
		{
			qDebug() << "Warning: Mesh face with not exactly 3 indices, ignoring this primitive.";
			continue;
		}
		ret->indices[3*i] = face->mIndices[0];
		ret->indices[3*i+1] = face->mIndices[1];
		ret->indices[3*i+2] = face->mIndices[2];
	}

	// material
	ret->material = m_materials.at(mesh->mMaterialIndex);

	
	// create a vertex buffer
	ret->createVertexBuffer();

	return ret;
}

void MeshImporter::processSkeleton( const aiScene *scene, aiNode *node, Bone *parent, Bone &currentBone )
{
	currentBone.m_boneName = node->mName.length != 0 ? node->mName.C_Str() : "";

	currentBone.m_localTransform = conv(&node->mTransformation);

	if(parent) parent->addChild(&currentBone);

	for (uint imesh = 0; imesh < node->mNumMeshes; ++imesh)
	{
		MeshData* mesh = m_meshes[node->mMeshes[imesh]];
		currentBone.m_meshes.push_back(*mesh);
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
