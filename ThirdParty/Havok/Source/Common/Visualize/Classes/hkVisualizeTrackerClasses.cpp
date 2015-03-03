/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Visualize/hkVisualize.h>
static const char s_libraryName[] = "hkVisualize";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkVisualizeRegister() {}

#include <Common/Visualize/Container/CommandStream/DebugCommands/hkDebugCommands.h>


// hkDebugCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkDebugCommand, s_libraryName)


// hkEmptyDebugCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkEmptyDebugCommand, s_libraryName)


// hkDebugLineCommand ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkDebugLineCommand, s_libraryName)

 // Skipping Class hkDebugCommandTypeDiscriminator< 0 > as it is a template


// hkDebugCommandProcessor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDebugCommandProcessor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDebugCommandProcessor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDebugCommandProcessor, s_libraryName, hkSecondaryCommandDispatcher)

#include <Common/Visualize/Process/hkDebugDisplayProcess.h>


// hkDebugDisplayProcess ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDebugDisplayProcess)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDebugDisplayProcess)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDebugDisplayProcess, s_libraryName, hkReferencedObject)

#include <Common/Visualize/Process/hkInspectProcess.h>


// hkInspectProcess ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkInspectProcess)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ObjectPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkInspectProcess)
    HK_TRACKER_MEMBER(hkInspectProcess, m_autoUpdateList, 0, "hkArray<hkInspectProcess::ObjectPair, hkContainerHeapAllocator>") // hkArray< struct hkInspectProcess::ObjectPair, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkInspectProcess, m_vdb, 0, "hkVisualDebugger*") // class hkVisualDebugger*
    HK_TRACKER_MEMBER(hkInspectProcess, m_cache, 0, "hkPlatformObjectWriter::Cache*") // class hkPlatformObjectWriter::Cache*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkInspectProcess, s_libraryName, hkReferencedObject)


// ObjectPair hkInspectProcess

HK_TRACKER_DECLARE_CLASS_BEGIN(hkInspectProcess::ObjectPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkInspectProcess::ObjectPair)
    HK_TRACKER_MEMBER(hkInspectProcess::ObjectPair, obj, 0, "void*") // const void*
    HK_TRACKER_MEMBER(hkInspectProcess::ObjectPair, klass, 0, "hkClass*") // const hkClass*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkInspectProcess::ObjectPair, s_libraryName)

#include <Common/Visualize/Process/hkMemorySnapshotProcess.h>


// hkMemorySnapshotProcess ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkMemorySnapshotProcess)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkMemorySnapshotProcess)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkMemorySnapshotProcess, s_libraryName, hkReferencedObject)

#include <Common/Visualize/Process/hkRemoteObjectProcess.h>


// hkRemoteObjectServerSideListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRemoteObjectServerSideListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRemoteObjectServerSideListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkRemoteObjectServerSideListener, s_libraryName)


// hkRemoteObjectClientSideListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRemoteObjectClientSideListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRemoteObjectClientSideListener)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkRemoteObjectClientSideListener, s_libraryName)


// hkRemoteObjectProcess ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkRemoteObjectProcess)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SendObjectType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkRemoteObjectProcess)
    HK_TRACKER_MEMBER(hkRemoteObjectProcess, m_listeners, 0, "hkArray<hkRemoteObjectServerSideListener*, hkContainerHeapAllocator>") // hkArray< class hkRemoteObjectServerSideListener*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkRemoteObjectProcess, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkRemoteObjectProcess, SendObjectType, s_libraryName)

#include <Common/Visualize/Process/hkStatisticsProcess.h>


// hkStatisticsProcess ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkStatisticsProcess)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkStatisticsProcess)
    HK_TRACKER_MEMBER(hkStatisticsProcess, m_contexts, 0, "hkArray<hkProcessContext*, hkContainerHeapAllocator>") // hkArray< class hkProcessContext*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkStatisticsProcess, m_infoBuffer, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkStatisticsProcess, m_strBuffer, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkStatisticsProcess, s_libraryName, hkReferencedObject)

#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>


// hkDisplaySerializeIStream ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplaySerializeIStream)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplaySerializeIStream)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplaySerializeIStream, s_libraryName, hkIArchive)

#include <Common/Visualize/Serialize/hkDisplaySerializeOStream.h>


// hkDisplaySerializeOStream ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplaySerializeOStream)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplaySerializeOStream)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplaySerializeOStream, s_libraryName, hkOArchive)

#include <Common/Visualize/Serialize/hkObjectSerializationManager.h>


// hkObjectSerializationManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectSerializationManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectSerializationManager)
    HK_TRACKER_MEMBER(hkObjectSerializationManager, m_streamWriter, 0, "hkStreamWriter*") // class hkStreamWriter*
    HK_TRACKER_MEMBER(hkObjectSerializationManager, m_ptrToTypeMap, 0, "hkPointerMap<hkUint64, hkUint64, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, hkUint64, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkObjectSerializationManager, m_dataReg, 0, "hkObjectSerializeRegistry") // class hkObjectSerializeRegistry
    HK_TRACKER_MEMBER(hkObjectSerializationManager, m_pendingRequests, 0, "hkPointerMap<hkUint64, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, hkInt32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkObjectSerializationManager, m_idToFixups, 0, "hkPointerMap<hkUint64, hkArray<hkObjectSerialize::GlobalFixup, hkContainerHeapAllocator>*, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, hkArray< struct hkObjectSerialize::GlobalFixup, struct hkContainerHeapAllocator >*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkObjectSerializationManager, m_liveIds, 0, "hkArray<hkUint64, hkContainerHeapAllocator>") // hkArray< hkUint64, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkObjectSerializationManager, s_libraryName)

#include <Common/Visualize/Serialize/hkObjectSerialize.h>


// hkObjectSerialize ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectSerialize)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LocalFixup)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(GlobalFixup)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkObjectSerialize, s_libraryName)


// LocalFixup hkObjectSerialize
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkObjectSerialize, LocalFixup, s_libraryName)


// GlobalFixup hkObjectSerialize
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkObjectSerialize, GlobalFixup, s_libraryName)

#include <Common/Visualize/Serialize/hkObjectSerializeRegistry.h>


// hkObjectSerializeRegistry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkObjectSerializeRegistry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkObjectSerializeRegistry)
    HK_TRACKER_MEMBER(hkObjectSerializeRegistry, m_idToObjectMap, 0, "hkPointerMap<hkUint64, void*, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, void*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkObjectSerializeRegistry, m_idToObjectSizeMap, 0, "hkPointerMap<hkUint64, hkUint32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, hkUint32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkObjectSerializeRegistry, m_specialClassClassIds, 0, "hkArray<hkUint64, hkContainerHeapAllocator>") // hkArray< hkUint64, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkObjectSerializeRegistry, s_libraryName, hkReferencedObject)

#include <Common/Visualize/Shape/hkDisplayAABB.h>


// hkDisplayAABB ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayAABB)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayAABB)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayAABB, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplayBox.h>


// hkDisplayBox ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayBox)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayBox)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayBox, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplayCapsule.h>


// hkDisplayCapsule ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayCapsule)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayCapsule)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayCapsule, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplayCone.h>


// hkDisplayCone ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayCone)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayCone)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayCone, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplayConvex.h>


// hkDisplayConvex ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayConvex)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayConvex)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayConvex, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplayCylinder.h>


// hkDisplayCylinder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayCylinder)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayCylinder)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayCylinder, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplayGeometry.h>


// hkDisplayGeometry ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayGeometry)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayGeometry)
    HK_TRACKER_MEMBER(hkDisplayGeometry, m_geometry, 0, "hkGeometry*") // struct hkGeometry*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkDisplayGeometry, s_libraryName, hkReferencedObject)

#include <Common/Visualize/Shape/hkDisplayGeometryTypes.h>

// None hkDisplayGeometryType
HK_TRACKER_IMPLEMENT_SIMPLE(hkDisplayGeometryType, s_libraryName)
#include <Common/Visualize/Shape/hkDisplayMesh.h>


// hkDisplayMesh ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayMesh)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayMesh)
    HK_TRACKER_MEMBER(hkDisplayMesh, m_mesh, 0, "hkMeshBody *") // class hkMeshBody *
    HK_TRACKER_MEMBER(hkDisplayMesh, m_meshAsTagfile, 0, "hkArray<char, hkContainerHeapAllocator>") // hkArray< char, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayMesh, s_libraryName, hkDisplayGeometry)


// hkForwardingDisplayGeometryBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkForwardingDisplayGeometryBuilder)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkForwardingDisplayGeometryBuilder)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkForwardingDisplayGeometryBuilder, s_libraryName, hkDisplayGeometryBuilder)

#include <Common/Visualize/Shape/hkDisplayPlane.h>


// hkDisplayPlane ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayPlane)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayPlane)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayPlane, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplaySemiCircle.h>


// hkDisplaySemiCircle ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplaySemiCircle)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplaySemiCircle)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplaySemiCircle, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplaySphere.h>


// hkDisplaySphere ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplaySphere)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplaySphere)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplaySphere, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplayTaperedCapsule.h>


// hkDisplayTaperedCapsule ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayTaperedCapsule)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayTaperedCapsule)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayTaperedCapsule, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Shape/hkDisplayWireframe.h>


// hkDisplayWireframe ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayWireframe)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayWireframe)
    HK_TRACKER_MEMBER(hkDisplayWireframe, m_lines, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDisplayWireframe, s_libraryName, hkDisplayGeometry)

#include <Common/Visualize/Type/hkKeyboard.h>


// hkKeyboard ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkKeyboard, s_libraryName)

// None hkKeyboardCommand
HK_TRACKER_IMPLEMENT_SIMPLE(hkKeyboardCommand, s_libraryName)
#include <Common/Visualize/hkCommandRouter.h>


// hkCommandRouter ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkCommandRouter)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkCommandRouter)
    HK_TRACKER_MEMBER(hkCommandRouter, m_commandMap, 0, "hkPointerMultiMap<hkUint8, hkProcess*>") // class hkPointerMultiMap< hkUint8, class hkProcess* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkCommandRouter, s_libraryName, hkReferencedObject)

#include <Common/Visualize/hkDebugDisplay.h>


// debugRenderNowCallbacks ::

HK_TRACKER_DECLARE_CLASS_BEGIN(debugRenderNowCallbacks)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(debugRenderNowCallbacks)
    HK_TRACKER_MEMBER(debugRenderNowCallbacks, m_tweakee, 0, "void*") // void*
    HK_TRACKER_MEMBER(debugRenderNowCallbacks, m_tweakeeClass, 0, "hkClass*") // const hkClass*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(debugRenderNowCallbacks, s_libraryName)


// debugRenderNow ::

HK_TRACKER_DECLARE_CLASS_BEGIN(debugRenderNow)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(debugRenderNow)
    HK_TRACKER_MEMBER(debugRenderNow, m_title, 0, "char*") // const char*
    HK_TRACKER_MEMBER(debugRenderNow, m_callbacks, 0, "debugRenderNowCallbacks*") // struct debugRenderNowCallbacks*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(debugRenderNow, s_libraryName)


// hkDebugDisplay ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDebugDisplay)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDebugDisplay)
    HK_TRACKER_MEMBER(hkDebugDisplay, m_debugDisplayHandlers, 0, "hkArray<hkDebugDisplayHandler*, hkContainerHeapAllocator>") // hkArray< class hkDebugDisplayHandler*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkDebugDisplay, m_arrayLock, 0, "hkCriticalSection*") // class hkCriticalSection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDebugDisplay, s_libraryName, hkReferencedObject)

#include <Common/Visualize/hkDebugDisplayHandler.h>

// hk.MemoryTracker ignore hkDebugDisplayHandler
#include <Common/Visualize/hkDisplayGeometryBuilder.h>


// hkDisplayGeometryBuilder ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDisplayGeometryBuilder)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDisplayGeometryBuilder)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkDisplayGeometryBuilder, s_libraryName, hkReferencedObject)

#include <Common/Visualize/hkDrawUtil.h>


// hkDrawUtil ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkDrawUtil)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DebugDisplayGeometrySettings)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkDrawUtil)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkDrawUtil, s_libraryName, hkReferencedObject)


// DebugDisplayGeometrySettings hkDrawUtil
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkDrawUtil, DebugDisplayGeometrySettings, s_libraryName)

#include <Common/Visualize/hkProcess.h>


// hkProcess ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkProcess)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ProcessType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkProcess)
    HK_TRACKER_MEMBER(hkProcess, m_inStream, 0, "hkDisplaySerializeIStream*") // class hkDisplaySerializeIStream*
    HK_TRACKER_MEMBER(hkProcess, m_outStream, 0, "hkDisplaySerializeOStream*") // class hkDisplaySerializeOStream*
    HK_TRACKER_MEMBER(hkProcess, m_displayHandler, 0, "hkDebugDisplayHandler*") // class hkDebugDisplayHandler*
    HK_TRACKER_MEMBER(hkProcess, m_processHandler, 0, "hkProcessHandler*") // class hkProcessHandler*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkProcess, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkProcess, ProcessType, s_libraryName)

#include <Common/Visualize/hkProcessContext.h>


// hkProcessContext ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkProcessContext)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkProcessContext)
    HK_TRACKER_MEMBER(hkProcessContext, m_owner, 0, "hkVisualDebugger*") // class hkVisualDebugger*
    HK_TRACKER_MEMBER(hkProcessContext, m_monitorStreamBegins, 0, "hkInplaceArray<char*, 6, hkContainerHeapAllocator>") // class hkInplaceArray< const char*, 6, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkProcessContext, m_monitorStreamEnds, 0, "hkInplaceArray<char*, 6, hkContainerHeapAllocator>") // class hkInplaceArray< const char*, 6, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT_BASE(hkProcessContext, s_libraryName)

#include <Common/Visualize/hkProcessFactory.h>


// hkProcessFactory ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkProcessFactory)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ProcessIdPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkProcessFactory)
    HK_TRACKER_MEMBER(hkProcessFactory, m_name2creationFunction, 0, "hkArray<hkProcessFactory::ProcessIdPair, hkContainerHeapAllocator>") // hkArray< struct hkProcessFactory::ProcessIdPair, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkProcessFactory, m_criticalSection, 0, "hkCriticalSection*") // class hkCriticalSection*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkProcessFactory, s_libraryName, hkReferencedObject)


// ProcessIdPair hkProcessFactory

HK_TRACKER_DECLARE_CLASS_BEGIN(hkProcessFactory::ProcessIdPair)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkProcessFactory::ProcessIdPair)
    HK_TRACKER_MEMBER(hkProcessFactory::ProcessIdPair, m_name, 0, "hkStringPtr") // hkStringPtr
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkProcessFactory::ProcessIdPair, s_libraryName)

#include <Common/Visualize/hkProcessHandler.h>

// hk.MemoryTracker ignore hkProcessHandler
#include <Common/Visualize/hkProcessRegisterUtil.h>


// hkProcessRegisterUtil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkProcessRegisterUtil, s_libraryName)

#include <Common/Visualize/hkServerDebugDisplayHandler.h>


// hkServerDebugDisplayHandler ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkServerDebugDisplayHandler)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(UnbuiltGeometryInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkServerDebugDisplayHandler)
    HK_TRACKER_MEMBER(hkServerDebugDisplayHandler, m_outstreamLock, 0, "hkCriticalSection*") // class hkCriticalSection*
    HK_TRACKER_MEMBER(hkServerDebugDisplayHandler, m_geometriesAwaitingRequests, 0, "hkArray<hkServerDebugDisplayHandler::UnbuiltGeometryInfo, hkContainerHeapAllocator>") // hkArray< struct hkServerDebugDisplayHandler::UnbuiltGeometryInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkServerDebugDisplayHandler, m_geometriesAwaitingDeparture, 0, "hkArray<hkServerDebugDisplayHandler::UnbuiltGeometryInfo, hkContainerHeapAllocator>") // hkArray< struct hkServerDebugDisplayHandler::UnbuiltGeometryInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkServerDebugDisplayHandler, m_continueData, 0, "hkReferencedObject*") // class hkReferencedObject*
    HK_TRACKER_MEMBER(hkServerDebugDisplayHandler, m_hashCountMap, 0, "hkPointerMap<hkUint64, hkInt32, hkContainerHeapAllocator>") // class hkPointerMap< hkUint64, hkInt32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkServerDebugDisplayHandler, s_libraryName, hkReferencedObject)


// UnbuiltGeometryInfo hkServerDebugDisplayHandler

HK_TRACKER_DECLARE_CLASS_BEGIN(hkServerDebugDisplayHandler::UnbuiltGeometryInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkServerDebugDisplayHandler::UnbuiltGeometryInfo)
    HK_TRACKER_MEMBER(hkServerDebugDisplayHandler::UnbuiltGeometryInfo, m_source, 0, "hkReferencedObject*") // const class hkReferencedObject*
    HK_TRACKER_MEMBER(hkServerDebugDisplayHandler::UnbuiltGeometryInfo, m_builder, 0, "hkDisplayGeometryBuilder*") // class hkDisplayGeometryBuilder*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkServerDebugDisplayHandler::UnbuiltGeometryInfo, s_libraryName)

#include <Common/Visualize/hkServerProcessHandler.h>


// hkServerProcessHandler ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkServerProcessHandler)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkServerProcessHandler)
    HK_TRACKER_MEMBER(hkServerProcessHandler, m_commandRouter, 0, "hkCommandRouter") // class hkCommandRouter
    HK_TRACKER_MEMBER(hkServerProcessHandler, m_processList, 0, "hkArray<hkProcess*, hkContainerHeapAllocator>") // hkArray< class hkProcess*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkServerProcessHandler, m_contexts, 0, "hkArray<hkProcessContext*, hkContainerHeapAllocator>") // hkArray< class hkProcessContext*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkServerProcessHandler, s_libraryName, hkReferencedObject)

#include <Common/Visualize/hkVersionReporter.h>


// hkVersionReporter ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVersionReporter, s_libraryName)

#include <Common/Visualize/hkVirtualFramebufferProtocol.h>

// hkVirtualFramebufferProtocol Version
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkVirtualFramebufferProtocol::Version, s_libraryName, hkVirtualFramebufferProtocol_Version)
// hkVirtualFramebufferProtocol Commands
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkVirtualFramebufferProtocol::Commands, s_libraryName, hkVirtualFramebufferProtocol_Commands)
// hkVirtualFramebufferProtocol FramebufferCommands
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkVirtualFramebufferProtocol::FramebufferCommands, s_libraryName, hkVirtualFramebufferProtocol_FramebufferCommands)
// hkVirtualFramebufferProtocol FramebufferFormat
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkVirtualFramebufferProtocol::FramebufferFormat, s_libraryName, hkVirtualFramebufferProtocol_FramebufferFormat)
#include <Common/Visualize/hkVirtualFramebufferServer.h>


// hkVirtualFramebufferRelativeRect ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVirtualFramebufferRelativeRect, s_libraryName)


// hkVirtualFramebuffer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVirtualFramebuffer)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DataType)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RowOrder)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DisplayRotation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVirtualFramebuffer)
    HK_TRACKER_MEMBER(hkVirtualFramebuffer, m_data, 0, "hkUint8*") // const hkUint8*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVirtualFramebuffer, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVirtualFramebuffer, DataType, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVirtualFramebuffer, RowOrder, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVirtualFramebuffer, DisplayRotation, s_libraryName)


// hkVirtualGamepad ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVirtualGamepad)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Stick)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Trigger)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Button)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkVirtualGamepad, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVirtualGamepad, Button, s_libraryName)


// Stick hkVirtualGamepad
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVirtualGamepad, Stick, s_libraryName)


// Trigger hkVirtualGamepad
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVirtualGamepad, Trigger, s_libraryName)

// hk.MemoryTracker ignore hkVirtualGamepadHandler

// hkVirtualKeyEvent ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVirtualKeyEvent, s_libraryName)

// hk.MemoryTracker ignore hkVirtualKeyEventHandler

// hkVirtualMouse ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVirtualMouse, s_libraryName)

// hk.MemoryTracker ignore hkVirtualMouseHandler

// hkVirtualFileDrop ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVirtualFileDrop)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVirtualFileDrop)
    HK_TRACKER_MEMBER(hkVirtualFileDrop, m_files, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVirtualFileDrop, s_libraryName)

// hk.MemoryTracker ignore hkVirtualFileDropHandler

// hkVirtualFramebufferServerClient ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVirtualFramebufferServerClient)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RuntimeOptions)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVirtualFramebufferServerClient)
    HK_TRACKER_MEMBER(hkVirtualFramebufferServerClient, m_socket, 0, "hkSocket*") // class hkSocket*
    HK_TRACKER_MEMBER(hkVirtualFramebufferServerClient, m_outStream, 0, "hkDisplaySerializeOStream*") // class hkDisplaySerializeOStream*
    HK_TRACKER_MEMBER(hkVirtualFramebufferServerClient, m_inStream, 0, "hkDisplaySerializeIStream*") // class hkDisplaySerializeIStream*
    HK_TRACKER_MEMBER(hkVirtualFramebufferServerClient, m_framebufferDiffStore, 0, "hkArray<hkUint8, hkContainerHeapAllocator>") // hkArray< hkUint8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVirtualFramebufferServerClient, s_libraryName)


// RuntimeOptions hkVirtualFramebufferServerClient
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkVirtualFramebufferServerClient, RuntimeOptions, s_libraryName)


// hkVirtualFramebufferServer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVirtualFramebufferServer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVirtualFramebufferServer)
    HK_TRACKER_MEMBER(hkVirtualFramebufferServer, m_gamepadHandlers, 0, "hkArray<hkVirtualGamepadHandler*, hkContainerHeapAllocator>") // hkArray< class hkVirtualGamepadHandler*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVirtualFramebufferServer, m_keyboardHandlers, 0, "hkArray<hkVirtualKeyEventHandler*, hkContainerHeapAllocator>") // hkArray< class hkVirtualKeyEventHandler*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVirtualFramebufferServer, m_mouseHandlers, 0, "hkArray<hkVirtualMouseHandler*, hkContainerHeapAllocator>") // hkArray< class hkVirtualMouseHandler*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVirtualFramebufferServer, m_filedropHandlers, 0, "hkArray<hkVirtualFileDropHandler*, hkContainerHeapAllocator>") // hkArray< class hkVirtualFileDropHandler*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVirtualFramebufferServer, m_clients, 0, "hkArray<hkVirtualFramebufferServerClient*, hkContainerHeapAllocator>") // hkArray< struct hkVirtualFramebufferServerClient*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVirtualFramebufferServer, m_server, 0, "hkSocket*") // class hkSocket*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkVirtualFramebufferServer, s_libraryName, hkReferencedObject)

#include <Common/Visualize/hkVisualDebugger.h>


// hkVisualDebuggerClient ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVisualDebuggerClient)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVisualDebuggerClient)
    HK_TRACKER_MEMBER(hkVisualDebuggerClient, m_socket, 0, "hkSocket*") // class hkSocket*
    HK_TRACKER_MEMBER(hkVisualDebuggerClient, m_processHandler, 0, "hkServerProcessHandler*") // class hkServerProcessHandler*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVisualDebuggerClient, s_libraryName)


// hkVisualDebuggerTrackedObject ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVisualDebuggerTrackedObject)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVisualDebuggerTrackedObject)
    HK_TRACKER_MEMBER(hkVisualDebuggerTrackedObject, m_ptr, 0, "void*") // void*
    HK_TRACKER_MEMBER(hkVisualDebuggerTrackedObject, m_class, 0, "hkClass*") // const hkClass*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkVisualDebuggerTrackedObject, s_libraryName)


// hkVdbChunk ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkVdbChunk, s_libraryName)


// hkVisualDebugger ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkVisualDebugger)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkVisualDebugger)
    HK_TRACKER_MEMBER(hkVisualDebugger, m_server, 0, "hkSocket*") // class hkSocket*
    HK_TRACKER_MEMBER(hkVisualDebugger, m_clients, 0, "hkArray<hkVisualDebuggerClient, hkContainerHeapAllocator>") // hkArray< struct hkVisualDebuggerClient, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVisualDebugger, m_contexts, 0, "hkArray<hkProcessContext*, hkContainerHeapAllocator>") // hkArray< class hkProcessContext*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVisualDebugger, m_trackedObjects, 0, "hkArray<hkVisualDebuggerTrackedObject, hkContainerHeapAllocator>") // hkArray< struct hkVisualDebuggerTrackedObject, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVisualDebugger, m_trackCallbacks, 0, "hkArray<void*, hkContainerHeapAllocator>") // hkArray< void*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVisualDebugger, m_trackCallbackHandles, 0, "hkArray<void*, hkContainerHeapAllocator>") // hkArray< void*, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVisualDebugger, m_classReg, 0, "hkVtableClassRegistry*") // const class hkVtableClassRegistry*
    HK_TRACKER_MEMBER(hkVisualDebugger, m_defaultProcesses, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVisualDebugger, m_requiredProcesses, 0, "hkArray<hkStringPtr, hkContainerHeapAllocator>") // hkArray< hkStringPtr, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkVisualDebugger, m_frameTimer, 0, "hkStopwatch") // class hkStopwatch
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkVisualDebugger, s_libraryName, hkReferencedObject)

#include <Common/Visualize/hkVisualDebuggerProtocol.h>

// hkVisualDebuggerProtocol ServerToClientCommands
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkVisualDebuggerProtocol::ServerToClientCommands, s_libraryName, hkVisualDebuggerProtocol_ServerToClientCommands)
// hkVisualDebuggerProtocol BidirectionalCommands
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkVisualDebuggerProtocol::BidirectionalCommands, s_libraryName, hkVisualDebuggerProtocol_BidirectionalCommands)
// hkVisualDebuggerProtocol ClientToServerCommands
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkVisualDebuggerProtocol::ClientToServerCommands, s_libraryName, hkVisualDebuggerProtocol_ClientToServerCommands)

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
