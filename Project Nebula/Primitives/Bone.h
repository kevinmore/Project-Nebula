#pragma once
#include <Primitives/MeshData.h>

class Bone
{
public:
	const char* m_boneName;
	Bone* m_parent;
	QVector<Bone*> m_children;
	mat4 m_localTransform;
	mat4 m_globalTransform;
	MeshData m_mesh;

	Bone(const char* boneName, Bone* parent, const mat4 &transform, MeshData &mesh)
	{
		// init the current bone
		m_boneName = boneName;
		m_parent = parent;
		m_localTransform = transform;
		m_mesh = mesh;
		// add the current bone to its parent if it's not the root
		if(parent) parent->addChild(this);
	}

	void addChild(Bone* child)
	{
		m_children.push_back(child);
	}

	QVector<Bone*> getChildren()
	{
		return m_children;
	}

	Bone* getChild(GLuint i)
	{
		if(m_children.isEmpty()) return NULL;
		else return m_children.at(i);
	}

	int childCount()
	{
		if(m_children.isEmpty()) return 0;
		else return m_children.size();
	}

	MeshData getMeshData()
	{
		return m_mesh;
	}

	static Bone* freeSkeleton(Bone* root)
	{
		if(!root) return NULL; // empty skeleton
		for (int i = 0; i < root->childCount(); ++i)
		{
			freeSkeleton(root->getChild(i));
		}
		free(root);

		return NULL;
	}

	// calculate and set the global transformation
	// for each bone in the skeleton
	static Bone* sortSkeleton(Bone* root)
	{
		if(!root) return NULL; // empty skeleton
		root->m_globalTransform = calcGlobalTransformation(root);
		for (int i = 0; i < root->childCount(); ++i)
		{
			sortSkeleton(root->getChild(i));
		}
		return NULL;
	}

	static mat4 calcGlobalTransformation(Bone* bone)
	{
		// empty bone
		mat4 result;
		if (!bone) return result;

		result = bone->m_localTransform;
		while (bone->m_parent)
		{
			// NOTE: the matrix multiplying order is very important!
			result = bone->m_parent->m_localTransform * result;
			bone = bone->m_parent;
		}
		return result;
	}

	// apply a transformation to each bone
	static Bone* configureSkeleton(Bone* root, mat4 &transform)
	{
		if(!root) return NULL; // empty skeleton
		// NOTE: the matrix multiplying order is very important!
		root->m_localTransform *= transform;
		for (int i = 0; i < root->childCount(); ++i)
		{
			configureSkeleton(root->getChild(i), transform);
		}
		return NULL;
	}

	static void printSkeleton(Bone* root)
	{
		if(!root) return; // empty skeleton
		qDebug()<<"Bone name: "<<root->m_boneName << " Global transform: "<<root->m_globalTransform;
		for (int i = 0; i < root->childCount(); ++i)
		{
			printSkeleton(root->getChild(i));
		}
	}

	~Bone(void);
};

