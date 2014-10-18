#include "MeshImporter.h"



MeshImporter::MeshImporter(void)
{
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

bool MeshImporter::loadMeshFromFile( const QString &fileName )
{
	Assimp::Importer importer;

	const aiScene* scene = importer.ReadFile(fileName.toStdString(),
		aiProcess_GenSmoothNormals |
		aiProcess_CalcTangentSpace |
		aiProcess_Triangulate |
		aiProcess_JoinIdenticalVertices |
		aiProcess_SortByPType
		);

	if (!scene)
	{
		qDebug() << "Error loading mesh file: " << importer.GetErrorString();
		return false;
	}

	// Materials must be loaded before meshes, and meshes must be loaded before bones.

	// load materials
	if (scene->HasMaterials())
	{
		for (unsigned int i = 0; i < scene->mNumMaterials; ++i)
		{
			MaterialInfo* mater = processMaterial(scene->mMaterials[i]);
			m_materials.push_back(mater);
		}
	}

	// load mesh
	if (scene->HasMeshes())
	{
		for (unsigned int ii = 0; ii < scene->mNumMeshes; ++ii)
		{
			m_meshes.push_back(processMesh(scene->mMeshes[ii]));
		}
	}
	else
	{
		qDebug() << "Error: No meshes found";
		return false;
	}

	// load bones
	if (scene->mRootNode != NULL)
	{
		Bone *root = new Bone();
		processSkeleton(scene, scene->mRootNode, 0, *root);
		m_Root = root;
	}
	else
	{
		qDebug() << "Error loading model";
		return false;
	}

	return true;

	return true;
}

MaterialInfo* MeshImporter::processMaterial( aiMaterial *material )
{
	MaterialInfo* mater(new MaterialInfo);
	aiString mname;
	material->Get(AI_MATKEY_NAME, mname);
	if (mname.length > 0)
		mater->Name = mname.C_Str();

	int shadingModel;
	material->Get(AI_MATKEY_SHADING_MODEL, shadingModel);

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

		material->Get(AI_MATKEY_COLOR_AMBIENT, amb);
		material->Get(AI_MATKEY_COLOR_DIFFUSE, dif);
		material->Get(AI_MATKEY_COLOR_SPECULAR, spec);
		material->Get(AI_MATKEY_SHININESS, shine);

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
	MeshData* newMesh(new MeshData);

	newMesh->meshName = mesh->mName.length != 0 ? mesh->mName.C_Str() : "";
	int indexOffset = m_indices.size();
	unsigned int indexCountBefore = m_indices.size();
	int vertindexoffset = m_vertices.size()/3;

	// Get Vertices
	if (mesh->mNumVertices > 0)
	{
		for (uint i = 0; i < mesh->mNumVertices; ++i)
		{
			aiVector3D &vec = mesh->mVertices[i];

			m_vertices.push_back(vec.x);
			m_vertices.push_back(vec.y);
			m_vertices.push_back(vec.z);
		}
	}

	// Get Normals
	if (mesh->HasNormals())
	{
		for (uint i = 0; i < mesh->mNumVertices; ++i)
		{
			aiVector3D &vec = mesh->mNormals[i];
			m_normals.push_back(vec.x);
			m_normals.push_back(vec.y);
			m_normals.push_back(vec.z);
		};
	}

	// Get mesh indexes
	for (uint i = 0; i < mesh->mNumFaces; ++i)
	{
		aiFace* face = &mesh->mFaces[i];
		if (face->mNumIndices != 3)
		{
			qDebug() << "Warning: Mesh face with not exactly 3 indices, ignoring this primitive.";
			continue;
		}

		m_indices.push_back(face->mIndices[0]+vertindexoffset);
		m_indices.push_back(face->mIndices[1]+vertindexoffset);
		m_indices.push_back(face->mIndices[2]+vertindexoffset);
	}

	newMesh->numIndices = m_indices.size() - indexCountBefore;
	newMesh->material = m_materials.at(mesh->mMaterialIndex);

	return newMesh;


}

void MeshImporter::processSkeleton( const aiScene *scene, aiNode *node, Bone *parentNode, Bone &newNode )
{
	newNode.m_boneName = node->mName.length != 0 ? node->mName.C_Str() : "";

	newNode.m_localTransform = conv(&node->mTransformation);
	newNode.m_meshes.resize(node->mNumMeshes);
	for (uint imesh = 0; imesh < node->mNumMeshes; ++imesh)
	{
		MeshData* mesh = m_meshes[node->mMeshes[imesh]];
		newNode.m_meshes.push_back(*mesh);
	}

	for (uint ich = 0; ich < node->mNumChildren; ++ich)
	{
		newNode.m_children.push_back(new Bone());
		processSkeleton(scene, node->mChildren[ich], parentNode, *newNode.m_children[ich]);
	}
}

