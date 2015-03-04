/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Compat/hkCompat.h>
#include <Common/Compat/Deprecated/Compat/hkCompatUtil.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Compat/Deprecated/Version/hkVersionRegistry.h>
#include <Common/Compat/Deprecated/Version/hkVersionUtilOld.h>
#include <Common/Serialize/Version/hkVersionUtil.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>
#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Compat/Deprecated/Compat/hkHavokAllClasses.h>

// This updater should never be called. It's just a stub because some old version
// code always expects to be able to update to the current version.
// We simply assert if ever called.

#define HK_COMPAT_VERSION_FROM hkHavok201311r1Classes
#define HK_COMPAT_VERSION_TO hkHavokCurrentClasses

namespace hkCompat_hk201311r1_hkCurrent
{
	static hkResult HK_CALL update(
								   hkArray<hkVariant>& objectsInOut,
								   hkObjectUpdateTracker& tracker )
	{
		HK_ERROR(0x13e06964, "Should never be called");
		return HK_FAILURE;
	}

	static const hkVersionRegistry::ClassAction s_updateActions[] =
	{
		{ 0, 0, 0, HK_NULL, HK_NULL }
	};
	static const hkVersionRegistry::ClassRename s_renames[] =
	{
		{ HK_NULL, HK_NULL }
	};

#define HK_COMPAT_OPTIONAL_UPDATE_FUNC update
#include<Common/Compat/Deprecated/Compat/hkCompat_Common.cxx>
#undef HK_COMPAT_VERSION_FROM
#undef HK_COMPAT_VERSION_TO

} // namespace hkCompat_hk201311r1_hkCurrent

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
