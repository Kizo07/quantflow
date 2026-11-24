// Source for northwind-10q-q3fy2026-excerpt.pdf (QuantFlow eval fixture).
// Rebuild: quarto typst compile northwind-10q-q3fy2026-excerpt.typ
// Northwind Instruments is fictional. All figures are invented for testing.
#set page(paper: "us-letter", margin: 0.9in, footer: align(center, text(7pt, fill: gray)[
  Synthetic document prepared for software testing. Northwind Instruments, Inc. is a fictional company; all figures are invented.
]))
#set text(font: "Libertinus Serif", size: 10pt)
#set par(justify: true)

#align(center)[
  #text(13pt, weight: "bold")[UNITED STATES SECURITIES AND EXCHANGE COMMISSION] \
  Washington, D.C. 20549 \
  #v(4pt)
  #text(12pt, weight: "bold")[FORM 10-Q] \
  QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 \
  For the quarterly period ended September 27, 2026 (third quarter of fiscal 2026) \
  #v(4pt)
  #text(12pt, weight: "bold")[NORTHWIND INSTRUMENTS, INC.] \
  (Excerpt: Part I, Item 2 and Part II, Item 1A)
]

#v(8pt)
== PART I, ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS (EXCERPT)

=== Segment results

Northwind reports two segments: *Test & Measurement* (bench and field instruments, calibration services) and *Industrial Sensing* (embedded pressure, flow and vibration sensors sold to OEMs). The table below compares the third quarter of fiscal 2026 with the immediately preceding quarter and with the third quarter of fiscal 2025. Dollars in millions.

#table(
  columns: (auto, 1fr, 1fr, 1fr),
  align: (left, right, right, right),
  stroke: 0.4pt,
  table.header([*Segment*], [*Q3 FY2026*], [*Q2 FY2026*], [*Q3 FY2025*]),
  [Test & Measurement revenue], [\$412.6], [\$398.1], [\$371.4],
  [Test & Measurement operating income], [\$95.3], [\$97.5], [\$84.7],
  [Test & Measurement operating margin], [23.1%], [24.5%], [22.8%],
  [Industrial Sensing revenue], [\$268.9], [\$281.7], [\$259.0],
  [Industrial Sensing operating income], [\$29.6], [\$39.4], [\$34.2],
  [Industrial Sensing operating margin], [11.0%], [14.0%], [13.2%],
  [Total revenue], [\$681.5], [\$679.8], [\$630.4],
  [Total segment operating income], [\$124.9], [\$136.9], [\$118.9],
)

*Test & Measurement.* Revenue rose 3.6% sequentially on higher calibration-services volume and a large order from a defense laboratory. Operating margin declined 140 basis points from the prior quarter, mainly because of a less favorable product mix (lower-margin field units) and \$4.1 million of expedited freight to meet the defense delivery schedule.

*Industrial Sensing.* Revenue fell 4.5% sequentially as two automotive OEM customers reduced build schedules. Operating margin declined 300 basis points from the prior quarter. The decline reflects lower absorption of fixed manufacturing costs at our Penang facility, a \$3.2 million inventory write-down of legacy vibration-sensor components, and higher prices for specialty silicon wafers.

*Outlook commentary in MD&A.* We expect Industrial Sensing build schedules to remain soft through the first half of fiscal 2027. We have begun consolidating two sensor lines into the Penang facility and expect related restructuring charges of \$6 million to \$9 million over the next three quarters.

== PART II, ITEM 1A. RISK FACTORS

Except as set out below, there have been no material changes to the risk factors disclosed in Part I, Item 1A of our Annual Report on Form 10-K for fiscal 2025 and in our Quarterly Report on Form 10-Q for the second quarter of fiscal 2026.

*New risk factor: Export controls on advanced measurement equipment.* In August 2026 the U.S. Department of Commerce proposed rules that would require licenses for exports of certain high-frequency signal analyzers to several jurisdictions. Sales of affected products to those jurisdictions were approximately 7% of Test & Measurement revenue in fiscal 2025. If the rules are adopted as proposed, we may be unable to obtain licenses on a timely basis, and customers may shift purchases to non-U.S. competitors.

*Updated risk factor: Customer concentration in Industrial Sensing.* We previously disclosed that our two largest Industrial Sensing customers accounted for approximately 28% of segment revenue. For the first nine months of fiscal 2026 these two customers accounted for approximately 34% of segment revenue. Both customers have recently reduced build schedules, and further reductions or the loss of either customer would materially reduce segment revenue and margins.

*Updated risk factor: Supply of specialty silicon wafers.* We previously described the risk that a single supplier provides most of our specialty silicon wafers. During the third quarter this supplier raised prices by approximately 12% and lengthened lead times to 26 weeks. We are qualifying a second supplier but do not expect qualification to be complete before the second half of fiscal 2027.

*Removed risk factor.* The risk factor titled "Integration of the Halvard Metrology acquisition" has been removed because integration was completed in the second quarter of fiscal 2026.
