/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Dynamics/hkpDynamics.h>
static const char s_libraryName[] = "hkpDynamicshkpPoweredChainData";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpDynamicshkpPoweredChainDataRegister() {}

#include <Physics2012/Dynamics/Constraint/Chain/Powered/hkpPoweredChainData.h>


// hkpPoweredChainData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPoweredChainData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPoweredChainData)
    HK_TRACKER_MEMBER(hkpPoweredChainData, m_atoms, 0, "hkpBridgeAtoms") // struct hkpBridgeAtoms
    HK_TRACKER_MEMBER(hkpPoweredChainData, m_infos, 0, "hkArray<hkpPoweredChainData::ConstraintInfo, hkContainerHeapAllocator>") // hkArray< struct hkpPoweredChainData::ConstraintInfo, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPoweredChainData, s_libraryName, hkpConstraintChainData)


// Runtime hkpPoweredChainData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPoweredChainData, Runtime, s_libraryName)


// ConstraintInfo hkpPoweredChainData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPoweredChainData::ConstraintInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPoweredChainData::ConstraintInfo)
    HK_TRACKER_MEMBER(hkpPoweredChainData::ConstraintInfo, m_motors, 0, "hkpConstraintMotor* [3]") // class hkpConstraintMotor* [3]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPoweredChainData::ConstraintInfo, s_libraryName)

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
