/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKGP_MESH_H
#define HKGP_MESH_H

#include <Common/Base/Config/hkProductFeatures.h>
#include <Geometry/Internal/hkcdInternal.h>
#include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>
#include <Common/Internal/GeometryProcessing/AbstractMesh/hkgpAbstractMesh.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Common/Base/Types/Geometry/hkStridedVertices.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

#if defined(HK_DEBUG) && defined(HK_PLATFORM_WIN32)
#undef HK_FORCE_INLINE
#define HK_FORCE_INLINE inline
#pragma auto_inline(off)
#endif

struct hkgpJobQueue;

//
// hkgpMeshBase
//
struct hkgpMeshBase
{	
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hkgpMeshBase);
	
	/// Vertex base type.
	struct BaseVertex
	{
		//+hk.MemoryTracker(ignore=True)
	};
	
	/// Vertex type.
	struct Vertex : public hkgpAbstractMeshDefinitions::Vertex<Vertex,BaseVertex>
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,Vertex);
		HKGP_FORCE_INLINE int		compare(const Vertex& v) const
		{
			for(int i=0;i<3;++i)
			{
				if(m_position(i)<v.m_position(i)) return(-1);
				if(m_position(i)>v.m_position(i)) return(+1);
			}
			return(0);
		}
		Vertex() : m_refs(-1),m_tag(-1) {}
		hkVector4	m_source;		///< Source position.
		hkVector4	m_position;		///< Current position.
		hkVector4	m_normal;		///< Normal (only valid if assignVertexNormal has been called).
		hkVector4	m_data;			///< User data.
		int			m_refs;			///< References count.
		int			m_tag;			///< User tag.
	};
	
	/// Triangle base type.
	struct BaseTriangle
	{
		//+hk.MemoryTracker(ignore=True)
	};
	
	/// Triangle type.
	struct Triangle : public hkgpAbstractMeshDefinitions::Triangle<Triangle,BaseTriangle,Vertex*>
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,Triangle);
		HKGP_FORCE_INLINE	hkVector4&			position(int i)			{ return(vertex(i)->m_position); }
		HKGP_FORCE_INLINE	const hkVector4&	position(int i) const	{ return(vertex(i)->m_position); }
		
		hkVector4	m_plane;	///< Triangle plane.
		int			m_partId;	///< Part to which this triangle belong to.
		int			m_planeId;	///< Plane index.
		int			m_material;	///< Material id.
		hkUlong		m_tag;		///< User tag.
		hkUlong		m_leafIdx;	///< Leaf index.
	};
	
	/// Edge type.
	struct Edge : public hkgpTopology::Edge<Edge,Vertex,Triangle>
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMeshBase::Edge);
		typedef hkgpTopology::Edge<Edge,Vertex,Triangle> tBase;
		Edge() : tBase() {}
		Edge(Triangle* t,unsigned e) : tBase(t,e) {}
		Edge(hkgpTopology::Uid id) : tBase(id) {}
		HKGP_FORCE_INLINE bool	hasTetrahedralTopology() const;
	};
};

///
/// hkgpMesh
///
class hkgpMesh : public hkgpAbstractMesh<hkgpMeshBase::Edge, hkgpMeshBase::Vertex, hkgpMeshBase::Triangle>
{
public:
	
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY);
	//
	// Enumerations
	//
	
	/// Handling of hollow parts WARNING: Keep in sync with CD counterpart.
	enum	eHollows
	{
		HOLLOW_KEEP		=	0,
		HOLLOW_MERGE	=	1,
		HOLLOW_DISCARD	=	2,
	};

	//
	// Types
	//

	/// Pair of triangle
	struct TrianglePair
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::TrianglePair);

		Triangle* m_a;
		Triangle* m_b;
	};
	
	/// SingleEdge
	struct SingleEdge
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::SingleEdge);

		const Vertex*	m_a;
		const Vertex*	m_b;
		Edge			m_e;
		int				m_refs;		
								SingleEdge() : m_a(0),m_b(0),m_e(0,0),m_refs(0)									{}
								SingleEdge(const Vertex* a,const Vertex* b) : m_a(a),m_b(b),m_e(0,0),m_refs(0)	{}
		HKGP_FORCE_INLINE int	hash() const;
		HKGP_FORCE_INLINE bool	operator==(const SingleEdge& x) const;
	};
	
	/// FloodPolicy
	struct FloodPolicy
	{
		HKGP_FORCE_INLINE bool canCrossEdge(hkgpMesh*,const Edge&) const { return(true); }
	};
	
	/// FloodFillDetachedPartsPolicy
	struct FloodFillDetachedPartsPolicy
	{
		HKGP_FORCE_INLINE bool canCrossEdge(hkgpMesh*,const Edge&) const { return(true); }
	};
	
	/// FloodFillDetachedOrMaterialBoundariesPartsPolicy
	struct FloodFillDetachedOrMaterialBoundariesPartsPolicy
	{
		HKGP_FORCE_INLINE bool canCrossEdge(hkgpMesh*,const Edge& edge) const { return edge.isNaked() || (edge.triangle()->m_material == edge.link().triangle()->m_material); }
	};
	
	/// CollapseEdgePolicy
	struct CollapseEdgePolicy
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::CollapseEdgePolicy);

		HKGP_FORCE_INLINE bool operator()(const Edge& e) const;
		hkgpMesh*	m_mesh;
		Vertex*		m_newVertex;
		bool		m_updatePlanes;
	};
	
	/// FanEnumPolicy
	struct FanEdgeCollector
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::FanEdgeCollector);

		HKGP_FORCE_INLINE bool	operator()(const Edge& e);
		HKGP_FORCE_INLINE bool	hasNakedEdges() const;
		HKGP_FORCE_INLINE bool	hasHeterogeneousPlaneIds() const;
		HKGP_FORCE_INLINE int		countPlanes() const;
		HKGP_FORCE_INLINE void	setVerticesTag(int id) const;
		HKGP_FORCE_INLINE void	incVerticesTag(int id) const;
		HKGP_FORCE_INLINE int		countVerticesTag(int id) const;
		hkInplaceArray<Edge,16>	m_edges;
	};
	
	/// CollapseMetric.
	/// Evaluate the cost of moving an edge.start() to m_position
	struct CollapseMetric
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::CollapseMetric);

		HKGP_FORCE_INLINE			CollapseMetric(hkgpMesh* mesh, hkVector4Parameter position);
		HKGP_FORCE_INLINE bool	operator()(Edge edge);
		hkgpMesh*	m_mesh;
		hkVector4	m_position;
		hkSimdReal	m_minDot;
		hkSimdReal	m_srcArea;
		hkSimdReal	m_dstArea;
	};
	
	/// SimplifyConfig
	struct SimplifyConfig
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::SimplifyConfig);

		hkSimdReal		m_maxCosAngle;
		hkSimdReal		m_minArea;
		SimplifyConfig()
		{
			m_maxCosAngle.setFromFloat(0.9999f);
			m_minArea.setFromFloat(0.0001f);
		}
	};
	
	/// SurfaceSamplingConfig
	struct SurfaceSamplingConfig
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::SurfaceSamplingConfig);
		hkSimdReal	m_offset;				///< Offset wrt. to surface where to generate samples.
		hkSimdReal	m_edgeSamplesSpacing;	///< Space between samples when sampling edges.
		hkSimdReal	m_maxCosAngle;			///< Edge angle greater than this will not be sampled.
		hkBool32	m_discardConvex;		///< Do not sample convex edges.
		hkBool32	m_doVertices;			///< Sample vertices.
		hkBool32	m_doEdges;				///< Sample edges.
		hkBool32	m_edgeCenterOnly;		///< Only generate edge center.
		SurfaceSamplingConfig()
		{
			m_offset				.   setZero();
			m_edgeSamplesSpacing	=	hkSimdReal_1;
			m_maxCosAngle			.   setFromFloat(0.9999f);
			m_discardConvex			=	true;
			m_doVertices			=	true;
			m_doEdges				=	true;
			m_edgeCenterOnly		=	false;
		}
	};
	
	/// HoleFillingConfig
	struct HoleFillingConfig
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::HoleFillingConfig);
		HoleFillingConfig()
		{

			m_fillRings			=	true;
			m_fillAFM			=	false;
		}



		hkBool32	m_fillRings;			///< Try to fill rings.
		hkBool32	m_fillAFM;				///< Try to fill holes using an advancing front method.
	};

	/// IConvexOverlap
	class IConvexOverlap
	{
		//+hk.MemoryTracker(ignore=True)
	public:
		class IConvexShape
		{
		public:
			virtual				~IConvexShape() {}
			virtual int			getMaxIndex() const { HK_ERROR(0xCFA7DFCC,"Not implemented");return(0); }
			virtual void		getSupportingVertex(hkVector4Parameter direction,hkVector4& output) const=0;
			virtual hkAabb		getBoundingBox() const=0;
		};
		virtual			~IConvexOverlap() {}
		virtual hkReal	distance(const IConvexShape* shapeA,const IConvexShape* shapeB,bool allowPenetration=true) const=0;
		virtual bool	checkOverlap(const IConvexShape* shapeA,const IConvexShape* shapeB,hkReal minDist=0) const=0;
	};
	
	/// PointShape
	class PointShape : public IConvexOverlap::IConvexShape
	{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::PointShape);

		PointShape(hkVector4Parameter x) : m_x(x) {}
		HKGP_FORCE_INLINE void		getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const 
		{ 
			out=m_x;
			out.setInt24W(0); 
		}
		HKGP_FORCE_INLINE hkAabb		getBoundingBox() const 
		{ 
			const hkSimdReal r = m_x.getW();
			
			hkAabb b;
			b.m_min.setSub(m_x,r);
			b.m_max.setAdd(m_x,r);
			b.m_min.zeroComponent<3>();
			b.m_max.zeroComponent<3>();
			return b;
		}
		hkVector4					m_x;		
	};
	
	/// LineShape
	class LineShape : public IConvexOverlap::IConvexShape
	{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::LineShape);

		LineShape(hkVector4Parameter a,hkVector4Parameter b, hkReal r=0) : m_a(a),m_b(b) { m_a.setInt24W(0);m_b.setInt24W(1);m_radius.setFromFloat(r); }
		
		HKGP_FORCE_INLINE void		getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const ;
		
		HKGP_FORCE_INLINE hkAabb		getBoundingBox() const 
		{ 
			hkAabb aabb;
			aabb.m_min.setMin(m_a,m_b);
			aabb.m_max.setMax(m_a,m_b);
			return aabb; 
		}
		hkVector4					m_a,m_b;
		hkSimdReal					m_radius;
	};
	
	/// ExtrudeShape
	class ExtrudeShape : public IConvexOverlap::IConvexShape
	{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::ExtrudeShape);

		HKGP_FORCE_INLINE ExtrudeShape(Triangle* t=0,hkReal width=0);
		HKGP_FORCE_INLINE void		getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const;
		HKGP_FORCE_INLINE hkAabb		getBoundingBox() const;
		hkInplaceArray<hkFourTransposedPoints, 2>	m_vertices;
	};
	
	/// TriangleShape
	struct TriangleShape : public IConvexOverlap::IConvexShape
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::TriangleShape);
		TriangleShape(Triangle* t=0) : IConvexOverlap::IConvexShape(),m_triangle(t) {}
		HKGP_FORCE_INLINE void		getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const;
		HKGP_FORCE_INLINE hkAabb		getBoundingBox() const;
		Triangle*					m_triangle;
	};
	
	/// ExternShape
	struct ExternShape : public IConvexOverlap::IConvexShape
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::ExternShape);
		ExternShape(const hkVector4* vertices,int numVertices) : IConvexOverlap::IConvexShape(),m_vertices(vertices),m_numVertices(numVertices) {}
		HKGP_FORCE_INLINE void		getSupportingVertex(hkVector4Parameter direction,hkVector4& out) const;
		HKGP_FORCE_INLINE hkAabb		getBoundingBox() const;
		const hkVector4*			m_vertices;
		int							m_numVertices;
	};
		
	/// ConvexHullShape
	struct ConvexHullShape : public IConvexOverlap::IConvexShape
	{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY,hkgpMesh::ConvexHullShape);
		ConvexHullShape(const hkgpConvexHull* ch=0) : IConvexOverlap::IConvexShape(),m_hull(ch) {}
		HKGP_FORCE_INLINE int			getMaxIndex() const { return(m_hull->getNumVertices()); }
		HKGP_FORCE_INLINE void		getSupportingVertex(hkVector4Parameter direction,hkVector4& output) const;
		HKGP_FORCE_INLINE hkAabb		getBoundingBox() const;
		const hkgpConvexHull*		m_hull;
	};	
		
	/// SortByArea
	struct SortByArea
	{
		HKGP_FORCE_INLINE bool operator()(const Triangle* a,const Triangle* b) const
		{
			// VS 2010 debug noSimd has issues with the code below.
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_DISABLED) && defined(_MSC_VER) && (_MSC_VER == 1600)
			const hkVector4 vAA = a->position(0);
			const hkVector4 vAB = a->position(1);
			const hkVector4 vAC = a->position(2);
			hkReal volatile areaA = hkGeometryProcessing::triangleArea2(vAA, vAB, vAC).getReal();

			const hkVector4 vBA = b->position(0);
			const hkVector4 vBB = b->position(1);
			const hkVector4 vBC = b->position(2);
			hkReal volatile areaB = hkGeometryProcessing::triangleArea2(vBA, vBB, vBC).getReal(); 
			return ( areaA > areaB );
#else
			
			return(	hkGeometryProcessing::triangleArea2(a->position(0),a->position(1),a->position(2))>
					hkGeometryProcessing::triangleArea2(b->position(0),b->position(1),b->position(2)));
#endif
		}
	};
	
	/// Location
	struct Location
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkgpMesh::Location);
		struct Region
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkgpMesh::Location::Region);
			enum eType { NONE,TRIANGLE,EDGE,VERTEX };
			HKGP_FORCE_INLINE				Region() : m_type(NONE),m_feature(Edge::null()) {}
			HKGP_FORCE_INLINE				Region(eType type, Edge feature) : m_type(type),m_feature(feature) {}
			HKGP_FORCE_INLINE	hkBool32	operator==(const Region& other) const
			{
				if(m_type!=other.m_type) return false;
				switch(m_type)
				{
					case	VERTEX:		return m_feature.start() == other.m_feature.start();
					case	EDGE:		return m_feature.master() == other.m_feature.master();
					case	TRIANGLE:	return m_feature.triangle() == other.m_feature.triangle();
					default:			return false;
				}
			}
			HKGP_FORCE_INLINE hkBool32	operator!=(const Region& other) const
			{
				return operator==(other)?0:1;
			}
			static hkBool32 HK_CALL		hasCommonSuperset(const Region*const* regions,int numRegions);
			eType	m_type;
			Edge	m_feature;			
		};
		Location()
		{
			m_normal.setZero();
			m_pseudoNormal.setZero();
			m_coordinates.setZero();
			m_squaredDistance = hkSimdReal_Max;
			setInside();
		}
		
		HKGP_FORCE_INLINE hkSimdReal signedDistance() const { return m_squaredDistance.sqrt() * m_inside; }
		HKGP_FORCE_INLINE hkVector4	signedNormal() const { hkVector4 n; n.setMul(m_inside,m_normal); return n; }
		HKGP_FORCE_INLINE hkBool32 isInside() const { return m_inside.isLessZero(); }
		HKGP_FORCE_INLINE hkBool32 isOutside() const { return m_inside.isGreaterZero(); }
		HKGP_FORCE_INLINE void setInside() { m_inside = hkSimdReal_Minus1; }
		HKGP_FORCE_INLINE void setOutside() { m_inside = hkSimdReal_1; }

		hkVector4	m_normal;
		hkVector4	m_pseudoNormal;
		hkVector4	m_coordinates;
		hkVector4	m_projection;
		hkSimdReal	m_squaredDistance;
		hkSimdReal	m_inside; // -1 == true ,  1 == false (= sign bit is true)
		Region		m_region;
	};
	
	//
	// Methods
	//

	//
	hkgpMesh();
	~hkgpMesh();
	void						reset();
	bool						isClosedManifold() const;
	hkAabb						getBoundingBox() const;
	bool						setPlane(const hkVector4& a,const hkVector4& b,const hkVector4& c,hkVector4& p,bool check=true);
	bool						setPlane(const Triangle* t,hkVector4& p,bool check=true);
	bool						updatePlane(Triangle* triangle);
	int							fetchPositions(hkArray<hkVector4>& positions) const;
	int							floodFillDetachedParts();
	int							floodFillDetachedOrMaterialBoundariesParts();
	int							countParts() const;
	int							explodeParts(hkArray<hkgpMesh*>& sets,bool doSimplify,bool doConvexHulls,bool sort=false, eHollows hollowPolicy = HOLLOW_KEEP) const;
	void						updatePlaneEquations();
	void						collapseEdge(Edge edge,bool updatePlanes);
	Edge						splitEdge(Edge edge,Vertex* vertex,bool updatePlanes);
	Edge						splitEdge(Edge edge,const hkVector4& position,bool updatePlanes);
	void						splitTriangle(Triangle* triangle,const hkVector4& position,bool updatePlanes);
	void						simplify(const SimplifyConfig& config=SimplifyConfig());
	void						locate(const hkVector4& x,Location& location, hkBool useTree=true) const;
	hkSimdReal					squaredDistanceToPoint(hkVector4Parameter x,hkVector4& normal,hkVector4Comparison& inside) const;
	hkgpConvexHull*				getConvexHull() const;
	hkSimdReal					getConvexHullVolume() const;
	hkgpMesh*					clone() const;
	void						append(hkgpMesh* other);
	void						invalidateConvexHull();
	bool						rebuildConvexHull();
	void						rebuildTrianglesTree(hkReal margin=0.0f);
	void						appendToGeometry(struct hkGeometry& geom,bool flipOrientation=false) const;
	void						appendFromGeometry(const struct hkGeometry& geom,const hkTransform& transform=hkTransform::getIdentity(),int partId=-1, bool flipOrientation=false, bool removeInvalidTriangles=true);
	void						appendFromGeometry(const struct hkGeometry& geom,const hkMatrix4& transform,int partId=-1, bool flipOrientation=false, bool removeInvalidTriangles=true);
	void						appendFromGeometryRaw(const struct hkGeometry& geom,const hkMatrix4& transform, bool flipOrientation=false);
	void						appendFromConvexHull(const hkgpConvexHull* hull);
	int							removeOrphanVertices();
	int							removeButterflies(int tag=-1);
	void						setPartId(int pid);
	void						setVerticesData(int index,hkReal value);
	void						setVerticesTag(int tag);
	void						copyPartIdToVerticesW();
	int							setPartIdAsConvexQuads(hkReal minCosAngle);
	void						assignVertexNormals();
	void						fetchAreaSortedTriangles(hkArray<Triangle*>& triangles,bool resetPlaneId=false) const;
	void						appendConcaveEdges(hkReal minConcavity,hkReal extrusion,const hkgpMesh* mesh);
	bool						checkOverlap(IConvexOverlap::IConvexShape* ishape,hkReal minDist=0,bool allowPenetration=true) const;
	bool						enumerateOverlaps(IConvexOverlap::IConvexShape* ishape,hkArray<Triangle*>& triangles,hkReal minDist=0,bool allowPenetration=true) const;
	bool						isConcave(Edge edge,hkReal eps=HK_REAL_EPSILON) const;
	static hkSimdReal HK_CALL	tetrahedronVolume6(Edge edge);
	Triangle*					createTriangle(Vertex* a,Vertex* b,Vertex* c);
	Edge						findEdge(const Vertex* s,const Vertex* e) const;
	Edge						flipEdge(Edge edge) const;
	int							buildPlaneIndices(hkReal minCosAngle,hkReal minDistance,bool stopAtMaterialBoundaries);
	void						setPartIdsAsPlaneIndices();
	static bool	HK_CALL			hasPosition(const Triangle* t,const hkVector4& position);
	void						removeTriangle(Triangle* t);
	void						sortTrianglesByArea(hkReal sign);
	hkSimdReal					projectPointOnSurface(hkReal offset,hkVector4& point,int iterations=16) const;
	hkSimdReal					projectPointOnSurface(hkReal offset,hkVector4Parameter normal, hkVector4& point,int iterations=16) const;
	bool						computeOffsetPoint(hkSimdRealParameter offset, hkVector4Parameter normal, hkVector4& point,int maxIterations=8, bool checkInside=true) const;
	void						generateSurfaceSamples(const SurfaceSamplingConfig& config,hkArray<hkVector4>& samplesOut, bool useWeight=false) const;
	void						generateConcaveEdges(hkReal offset,hkArray<hkVector4>& edgesOut, hkBool32 useWeight=false, hkgpJobQueue* jq=HK_NULL) const;
	void						generateEmptySpaceSamples(int maxOctreeDepth,hkReal scaling,hkArray<hkVector4>& samplesOut,bool negate=false) const;
	void						reportBadAreaTriangle(const hkVector4& a,const hkVector4& b,const hkVector4& c);
	void						reportBadEdgeLength(const hkVector4& a,const hkVector4& b);
	void						reportDuplicatedEdge(const hkVector4& a,const hkVector4& b);
	void						reportInvalidEdgeWinding(const hkVector4& a,const hkVector4& b);
	bool						hasValidTopology(bool raiseError=true) const;
	bool						bindEdge(Edge edge, bool onlyToNaked=true , bool mustMatch=true);
	void						bindEdgeList(hkArray<Edge>& edges);
	void						fillHoles(const HoleFillingConfig& config);
	void						fixTJunctions(hkReal maxDistanceToEdge, bool report=true);
	void						fixBindings();
	void						removePartTriangles(int id);
	void						removePlaneTriangles(int id);
	void						remapPlaneIndex(int from_id,int to_id);
	void						removePartFromClassification(int id,const hkTransform& transform, const hkgpMesh* classifier,bool removeInside);
	void						flipPartOrientation(int id);
	void						retriangulateFromPlanes();
	void						enumerateTriangleOverlaps(const hkVector4& a,const hkVector4& b,const hkVector4& c,hkArray<Triangle*>& triangles) const;
	hkBool32					checkTriangleOverlap(const hkVector4& a,const hkVector4& b,const hkVector4& c) const;
	int							extractAllLoops(hkArray<hkArray<Edge>*>& loops,int partId=-1) const;
	int							countFanPlanes(Edge edge) const;
	hkBool						checkEdgeCollapseConvex(const FanEdgeCollector& fan, Edge edge) const;
	hkBool						checkEdgeCollapseConvex(Edge edge) const;
	void						simplifyPlanes(bool report=false);
	void						initializeQEM();
	void						computeQEM(Edge e);
	void						simplifyQEM(int numTriangleToRemove, hkReal maxCost);
	void						applyLaplacianSmoothing(hkReal iterations);
	
	//
	// Inline's
	//
	
	static HKGP_FORCE_INLINE hkSimdReal	getTwiceArea(const Triangle* triangle);
	static HKGP_FORCE_INLINE hkVector4	getCentroid(const Triangle* triangle) { return hkGeometryProcessing::computeMedian(triangle->position(0), triangle->position(1), triangle->position(2)); }
	
	template <typename T>
	HKGP_FORCE_INLINE int			floodFill(T& boundaries);
	
	//
	// Fields
	//	
	const IConvexOverlap*		m_iconvexoverlap;			///< Convex overlap interface.
	void*						m_trianglesTree;			///< Triangles tree.
	mutable hkArray<hkVector4>	m_randSamples;				///< Precomputed random samples.
	hkArray<hkVector4>			m_planes;					///< Planes.
	hkArray<Triangle*>			m_planeRoots;				///< Plane roots (used to build planes indices).
	hkgpConvexHull*				m_convexHull;				///< Convex hull of the mesh.
	bool						m_hasErrors;				///< The mesh has error(s).
	bool						m_hasPerVertexNormals;		///< The mesh has vertex normal assigned.
	hkReal						m_epsMinEdgeSqLength;		///< Minimum allowed edge length squared.
	hkReal						m_epsMinTwiceTriangleArea;	///< Minimum allowed triangle area.
};

#include <Common/Internal/GeometryProcessing/Mesh/hkgpMesh.inl>

#undef TREEINT

#endif // HKGP_MESH_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
