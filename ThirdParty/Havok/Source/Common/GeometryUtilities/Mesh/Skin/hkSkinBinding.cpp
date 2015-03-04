/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Skin/hkSkinBinding.h>

//
//	Constructor

hkSkinBinding::hkSkinBinding(hkMeshShape* skinnedMesh, const hkMatrix4* HK_RESTRICT worldFromBoneTransforms, const hkStringPtr* HK_RESTRICT boneNames, int numBones)
:	hkMeshShape()
,	m_skin(skinnedMesh)
{
	m_worldFromBoneTransforms.append(worldFromBoneTransforms, numBones);
	m_boneNames.append(boneNames, numBones);
}

//
//	Serialization constructor

hkSkinBinding::hkSkinBinding(hkFinishLoadedObjectFlag flag)
:	hkMeshShape(flag)
,	m_skin(flag)
,	m_worldFromBoneTransforms(flag)
,	m_boneNames(flag)
{}

//
//	Returns the amount of sections

int hkSkinBinding::getNumSections() const
{
	return m_skin->getNumSections();
}

//
//	Gets information about a section

void hkSkinBinding::lockSection(int sectionIndex, hkUint8 accessFlags, hkMeshSection& sectionOut) const
{
	m_skin->lockSection(sectionIndex, accessFlags, sectionOut);
}

//
//	Unlocks a mesh section. Must be given exactly the same structure contents as was returned from a lockSection
//	otherwise behavior is undefined.

void hkSkinBinding::unlockSection(const hkMeshSection& section) const
{
	m_skin->unlockSection(section);
}

//
//	Returns an optional name of the mesh shape

const char* hkSkinBinding::getName() const
{
	return m_skin->getName();
}

//
//	Sets an optional name

void hkSkinBinding::setName(const char* n)
{
	m_skin->setName(n);
}

//
//	Returns the class type

const hkClass* hkSkinBinding::getClassType() const
{
	return &hkSkinBindingClass;
}

//
//	Returns the index of the bone with the given name

int hkSkinBinding::findBone(const char* name) const
{
	for (int k = m_boneNames.getSize() - 1 ; k >= 0; k--)
	{
		const char* boneName = m_boneNames[k].cString();

		if ( boneName && name && (hkString::strCmp(boneName, name) == 0) )
		{
			return k;
		}
	}

	return -1;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
