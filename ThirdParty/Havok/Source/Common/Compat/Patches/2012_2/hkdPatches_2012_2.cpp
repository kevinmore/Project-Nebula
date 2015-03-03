/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Compat/hkCompat.h>
#include <Common/Serialize/Data/hkDataObject.h>
#include <Common/Serialize/Version/hkVersionPatchManager.h>
#include <Common/Base/KeyCode.h>

// Registration function is at the end of the file

static void hkdDecorateFractureFaceActionGlobalDecorationData_2_to_3(hkDataObject& obj)
{
	hkDataArray src = obj["old_rawTransforms"].asArray();
	hkDataArray dst = obj["rawTransforms"].asArray();

	const int size = src.getSize();
	dst.setSize(size*8);
	for (int i = 0; i < size; i++)
	{
		hkDataObject qt = src[i].asObject();
		const hkQuaternion& quat = qt["rotation"].asQuaternion();
		const hkVector4& tran = qt["translation"].asVector4();

		dst[i*8  ] = hkFloat32(quat.m_vec(0));
		dst[i*8+1] = hkFloat32(quat.m_vec(1));
		dst[i*8+2] = hkFloat32(quat.m_vec(2));
		dst[i*8+3] = hkFloat32(quat.m_vec(3));
		dst[i*8+4] = hkFloat32(tran(0));
		dst[i*8+5] = hkFloat32(tran(1));
		dst[i*8+6] = hkFloat32(tran(2));
		dst[i*8+7] = hkFloat32(tran(3));
	}
}

void HK_CALL registerDestructionPatches_2012_2(hkVersionPatchManager& man)
{
#	define HK_PATCHES_FILE <Common/Compat/Patches/2012_2/hkdPatches_2012_2.cxx>
#	include <Common/Serialize/Version/hkVersionPatchManager.cxx>
#	undef HK_PATCHES_FILE
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
